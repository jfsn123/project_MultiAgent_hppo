# -*- coding: utf-8 -*-
import glob
import json
import math
import os
import re
from typing import Optional

import numpy as np
import torch

from python_scripts.PPO.utils.Project_config import gps_goal, path_list
from python_scripts.PPO.hppo.hppo_01 import HPPO as hppo

from python_scripts.PPO.utils.Webots_interfaces import Environment


def validate_and_clean_data(data, default_value=0.0):
    if isinstance(data, (list, tuple)):
        return [validate_and_clean_data(x, default_value) for x in data]
    if isinstance(data, np.ndarray):
        return np.nan_to_num(data, nan=default_value, posinf=default_value, neginf=-default_value)
    if isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return default_value
        return data
    return data


def _safe_torch_load(ckpt_path: str):
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.load(ckpt_path, map_location=map_location)


def _strip_common_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "policy.",
        "policy_catch.",
        "actor.",
        "actor_net.",
        "model.",
        "net.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _normalize_state_dict(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        normalized[_strip_common_prefixes(key)] = value
    return normalized


def _filter_compatible_state_dict(model, state_dict: dict) -> dict:
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if hasattr(value, "shape") and hasattr(model_state[key], "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            continue
        compatible[key] = value
    return compatible


def _all_catch_ckpts(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return []

    def _num(file_path: str) -> int:
        match = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return sorted((f for f in files if _num(f) >= 0), key=_num)


def _load_catch_model(agent, model_path: str) -> None:
    ckpt = _safe_torch_load(model_path)
    if isinstance(ckpt, dict):
        if "policy" in ckpt:
            state_dict = ckpt["policy"]
        elif "policy_hppo" in ckpt:
            state_dict = ckpt["policy_hppo"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        agent.policy.load_state_dict(state_dict)
        return
    except Exception:
        pass

    normalized_state_dict = _normalize_state_dict(state_dict)
    try:
        agent.policy.load_state_dict(normalized_state_dict)
        return
    except Exception:
        pass

    compatible_state_dict = _filter_compatible_state_dict(agent.policy, normalized_state_dict)
    if compatible_state_dict:
        agent.policy.load_state_dict(compatible_state_dict, strict=False)
        return

    agent.policy.load_state_dict(normalized_state_dict, strict=False)


def _clear_runtime_cache(images: Optional[list] = None) -> None:
    if images is not None:
        images.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _catch_distance_to_goal(env) -> float:
    try:
        gps_values = validate_and_clean_data(env.print_gps())
        gps1 = gps_values[0] if len(gps_values) > 0 else [0.0, 0.0, 0.0]
        x = float(gps1[1]) if len(gps1) > 1 else 0.0
        y = float(gps1[2]) if len(gps1) > 2 else 0.0
        dx = float(gps_goal[0]) - x
        dy = float(gps_goal[1]) - y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float("inf")


def _left_grasp_success(env) -> bool:
    vals = [
        env.darwin.get_touch_sensor_value("grasp_L1"),
        env.darwin.get_touch_sensor_value("grasp_L1_1"),
        env.darwin.get_touch_sensor_value("grasp_L1_2"),
    ]
    return any(v > 0.0 for v in vals)


def _right_grasp_success(env) -> bool:
    vals = [
        env.darwin.get_touch_sensor_value("grasp_R1"),
        env.darwin.get_touch_sensor_value("grasp_R1_1"),
        env.darwin.get_touch_sensor_value("grasp_R1_2"),
    ]
    return any(v > 0.0 for v in vals)


def _catch_success(env, distance_threshold: float = 0.04) -> bool:
    return _left_grasp_success(env) and _right_grasp_success(env) and _catch_distance_to_goal(env) <= distance_threshold


def _run_one_episode(env, catch_agent, episode_index: int, max_steps_per_episode: int, success_distance_threshold: float) -> bool:
    imgs = []
    steps = 0
    ok = False

    _, obs_tensor = env.get_img(steps)
    raw_robot_state = env.get_robot_state()
    robot_state = env.validate_and_clean_data(raw_robot_state)
    prev_discrete_actions = [1.0, 1.0]
    catch_state = env.build_catch_feature_state(
        robot_state=robot_state,
        prev_robot_state=None,
        prev_exec_actions=[0.0, 0.0],
        prev_discrete_actions=prev_discrete_actions,
    )
    prev_shoulder_action = 0.0
    prev_arm_action = 0.0

    while steps < max_steps_per_episode:
        action_dict = catch_agent.choose_action((obs_tensor, catch_state), greedy=True)

        d_action = action_dict["discrete_action"]
        d0 = int(d_action[0])
        d1 = int(d_action[1])
        action_shoulder = float(action_dict["continuous_action"][0].item())
        action_arm = float(action_dict["continuous_action"][1].item())
        masked_shoulder = prev_shoulder_action if d0 == 0 else action_shoulder
        masked_arm = prev_arm_action if d1 == 0 else action_arm

        gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()
        catch_flag = 1.0 if steps >= max_steps_per_episode - 1 else 0.0
        img_name = f"catch_ground_eval_{episode_index}_{steps}.png"

        print(
            f"第{episode_index}周期，第{steps}步 "
            f"离散数值：({d0},{d1}), 连续动作：{masked_shoulder:.4f},{masked_arm:.4f})"
        )
        print(f"[抓取阶段-动作前] left_gps1: {gps1}")

        _, done, _, _, _ = env.step(robot_state, masked_shoulder, masked_arm, steps, catch_flag)

        prev_shoulder_action = masked_shoulder
        prev_arm_action = masked_arm

        if _catch_success(env, success_distance_threshold):
            ok = True
            print(f"✅ 抓取成功，当前距离: {_catch_distance_to_goal(env):.4f}")
            env.wait(100)
            break

        steps += 1
        if done == 1 or catch_flag == 1.0:
            break

        _, next_obs_tensor = env.get_img(steps)
        obs_tensor = next_obs_tensor
        prev_robot_state = robot_state
        robot_state = env.get_robot_state()
        robot_state = env.validate_and_clean_data(robot_state)
        catch_state = env.build_catch_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[masked_shoulder, masked_arm],
            prev_discrete_actions=[float(d0), float(d1)],
        )

    if not ok:
        print(f"⚠️ 抓取未成功，当前距离: {_catch_distance_to_goal(env):.4f}")

    _clear_runtime_cache(imgs)
    return ok


def evaluate_one_catch_model(
    env,
    catch_agent,
    episodes: int = 100,
    max_steps_per_episode: int = 20,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
):
    success_count = 0

    for episode_index in range(1, int(episodes) + 1):
        print("\n==============================")
        print(f"🌍 Total Episode {episode_index}")
        print("==============================")

        env.reset()
        env.wait(reset_wait_ms)

        catch_ok = _run_one_episode(
            env=env,
            catch_agent=catch_agent,
            episode_index=episode_index,
            max_steps_per_episode=max_steps_per_episode,
            success_distance_threshold=success_distance_threshold,
        )

        success_count += int(catch_ok)
        running_rate = success_count / episode_index

        print("[Episode Result]")
        print(f"  - catch_success: {bool(catch_ok)}")
        print(f"  - running_success_rate: {success_count}/{episode_index} = {running_rate:.4f}")

        env.reset()
        env.wait(reset_wait_ms)

    return success_count / float(episodes) if episodes > 0 else 0.0


def run_batch_catch_models(
    robot,
    catch_dir: Optional[str] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 20,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
    save_json_path: Optional[str] = None,
):
    if catch_dir is None:
        catch_dir = path_list["model_path_catch_PPO_h"]

    catch_ckpts = _all_catch_ckpts(catch_dir)
    if not catch_ckpts:
        raise FileNotFoundError(f"no catch model checkpoint found in: {catch_dir}")

    env = Environment(robot)
    results = []

    for model_path in catch_ckpts:
        catch_agent = hppo(num_servos=2, state_dim=19)
        _load_catch_model(catch_agent, model_path)

        print("=" * 80)
        print(f"start evaluating catch model: {model_path}")

        success_rate = evaluate_one_catch_model(
            env=env,
            catch_agent=catch_agent,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            success_distance_threshold=success_distance_threshold,
            reset_wait_ms=reset_wait_ms,
        )

        results.append(
            {
                "model_path": model_path,
                "episodes": int(episodes),
                "success_rate": float(success_rate),
            }
        )
        print(f"[catch-ground-eval] model done: success_rate={success_rate:.4f}")

        # 每个模型完成后立即保存
        if save_json_path is None:
            test_log_dir = os.path.join(os.path.dirname(os.path.dirname(path_list["tai_log_path_PPO"])), "test_log")
            os.makedirs(test_log_dir, exist_ok=True)
            save_json_path = os.path.join(test_log_dir, "catch_ground_success_rates.json")

        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存 {save_json_path}")

    print("=" * 80)
    print(f"all catch models evaluated, results saved to: {save_json_path}")
    return results


def ground_catch_models_test(robot):
    """Direct-call interface: evaluate all catch models for 100 episodes each."""
    return run_batch_catch_models(
        robot=robot,
        episodes=100,
    )
