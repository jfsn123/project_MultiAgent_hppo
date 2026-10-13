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
from python_scripts.PPO.hppo.hppo import HPPO as d_hppo
from python_scripts.PPO.test.full_climb_test import (
    _latest_ckpt,
    _latest_upper_ckpt,
    _load_policy,
    _replay_upper_right_grasp,
    _restore_base_targets,
    _run_full_step,
    _run_upper_left_grasp,
    _set_upper_phase_targets,
)

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


def _latest_catch_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None

    def _num(file_path: str) -> int:
        match = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return max(files, key=_num)


def _all_decision_ckpts(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return []

    def _num(file_path: str) -> int:
        match = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return sorted((f for f in files if _num(f) >= 0), key=_num)


def _resolve_catch_model_path(model_path: Optional[str], catch_dir: str) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"catch model not found: {model_path}")
        return model_path

    latest = _latest_catch_ckpt(catch_dir)
    if not latest:
        raise FileNotFoundError(f"no catch model checkpoint found in: {catch_dir}")
    return latest


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


def _load_decision_model(agent, model_path: str) -> None:
    ckpt = _safe_torch_load(model_path)
    if isinstance(ckpt, dict):
        if "policy" in ckpt:
            state_dict = ckpt["policy"]
        elif "policy_decision" in ckpt:
            state_dict = ckpt["policy_decision"]
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
        foot_x = float(gps1[1]) if len(gps1) > 1 else 0.0
        foot_y = float(gps1[2]) if len(gps1) > 2 else 0.0
        dx = float(gps_goal[0]) - foot_x
        dy = float(gps_goal[1]) - foot_y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float("inf")


def _left_grasp_success(env) -> bool:
    left_vals = [
        env.darwin.get_touch_sensor_value("grasp_L1"),
        env.darwin.get_touch_sensor_value("grasp_L1_1"),
        env.darwin.get_touch_sensor_value("grasp_L1_2"),
    ]
    return any(v > 0.0 for v in left_vals)


def _right_grasp_success(env) -> bool:
    right_vals = [
        env.darwin.get_touch_sensor_value("grasp_R1"),
        env.darwin.get_touch_sensor_value("grasp_R1_1"),
        env.darwin.get_touch_sensor_value("grasp_R1_2"),
    ]
    return any(v > 0.0 for v in right_vals)


def _catch_success(env, distance_threshold: float = 0.04) -> bool:
    return _left_grasp_success(env) and _right_grasp_success(env) and _catch_distance_to_goal(env) <= distance_threshold


def _get_decision_state(env):
    pressure_state, _, pressure_detected = env.get_hand_pressure_state()
    decision_state = np.concatenate((
        np.asarray(pressure_state, dtype=np.float32),
        np.array([env.hand_foot_gps_distance(), env.get_foot_y_post()], dtype=np.float32),
    ))
    return decision_state, bool(pressure_detected)


def _judge_climb_decision(decision: int, decision_state, pressure_detected: bool):
    """Use the same three-stage target rule as PPO_episoid_0."""
    hand_foot_distance = float(decision_state[-2])
    foot_y_post = float(decision_state[-1])
    if not pressure_detected:
        expected = 0
    elif hand_foot_distance < 0.3 and foot_y_post > 0.03:
        expected = 2
    else:
        expected = 1
    labels = {0: "ground grasp", 1: "stepping", 2: "upper grasp"}
    return int(decision) == expected, f"expected {expected} ({labels[expected]})"


def _yes_no(flag: bool) -> str:
    return "yes" if bool(flag) else "no"


def _one_episode(env, catch_agent, episode_index: int, max_steps_per_episode: int, success_distance_threshold: float) -> bool:
    imgs = []
    steps = 0
    episode_success = False

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
        action_shoulder = float(action_dict["continuous_action"][0].item())
        action_arm = float(action_dict["continuous_action"][1].item())

        d0 = int(d_action[0])
        d1 = int(d_action[1])
        masked_shoulder = prev_shoulder_action if d0 == 0 else action_shoulder
        masked_arm = prev_arm_action if d1 == 0 else action_arm

        gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()
        catch_flag = 1.0 if steps >= max_steps_per_episode - 1 else 0.0
        img_name = f"decision_test_{episode_index}_{steps}.png"

        print(
            f"第{episode_index}周期，第{steps}步 "
            f"离散数值：({d0},{d1}), 连续动作：{masked_shoulder:.4f},{masked_arm:.4f})"
        )
        print(f"[抓取阶段-动作前] left_gps1: {gps1}")

        _, done, good, _, _ = env.step(robot_state, masked_shoulder, masked_arm, steps, catch_flag)

        prev_shoulder_action = masked_shoulder
        prev_arm_action = masked_arm

        if _catch_success(env, success_distance_threshold):
            episode_success = True
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

    _clear_runtime_cache(imgs)
    if not episode_success:
        print(f"⚠️ 抓取未成功，当前距离: {_catch_distance_to_goal(env):.4f}")
    return episode_success


def run(
    robot,
    model_path: Optional[str] = None,
    episodes: int = 100,
    max_steps_per_episode: int = 20,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
):
    """Test the catch model for a fixed number of episodes and report success rate."""
    env = Environment(robot)
    catch_agent = hppo(num_servos=2, state_dim=19)

    resolved_model_path = _resolve_catch_model_path(model_path, path_list["model_path_catch_PPO_h"])
    print(f"loading catch model: {resolved_model_path}")
    _load_catch_model(catch_agent, resolved_model_path)
    print(f"start catch evaluation for {episodes} episodes")

    success_count = 0
    for episode_index in range(1, int(episodes) + 1):
        print("=" * 60)
        print(f"episode {episode_index}/{episodes}")
        env.reset()
        env.wait(reset_wait_ms)
        episode_success = _one_episode(
            env=env,
            catch_agent=catch_agent,
            episode_index=episode_index,
            max_steps_per_episode=max_steps_per_episode,
            success_distance_threshold=success_distance_threshold,
        )
        success_count += int(episode_success)
        print(f"running success rate: {success_count}/{episode_index} = {success_count / episode_index:.4f}")

        env.reset()
        env.wait(reset_wait_ms)

    final_rate = success_count / float(episodes) if episodes > 0 else 0.0
    print("=" * 60)
    print(f"final success rate: {success_count}/{episodes} = {final_rate:.4f}")
    return final_rate


def evaluate_one_decision_model(
    env,
    decision_agent,
    catch_agent,
    tai_agent,
    upper_agent,
    episodes: int = 100,
    max_steps_per_episode: int = 20,
    upper_max_steps: int = 30,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
):
    """Run ground grasp, stepping and upper grasp, scoring decisions only."""
    stage_stats = {
        "ground_grasp": [0, 0],
        "stepping": [0, 0],
        "upper_grasp": [0, 0],
    }

    def check_decision(stage_name: str):
        state, pressure_detected = _get_decision_state(env)
        decision = int(decision_agent.choose_action(obs=state, greedy=True)["discrete_action"][0])
        ok, reason = _judge_climb_decision(decision, state, pressure_detected)
        stage_stats[stage_name][1] += 1
        stage_stats[stage_name][0] += int(ok)
        print(
            f"[Decision:{stage_name}] pressure={_yes_no(pressure_detected)}, "
            f"output={decision}, result={'correct' if ok else 'wrong'}, {reason}"
        )
        return decision, bool(ok)

    for episode_index in range(1, int(episodes) + 1):
        print(f"\n========== Decision Episode {episode_index}/{episodes} ==========")
        _restore_base_targets()
        env.reset()
        env.wait(reset_wait_ms)

        ground_decision, ground_ok = check_decision("ground_grasp")
        if not ground_ok or ground_decision != 0:
            continue
        ground_action_ok = _one_episode(
            env=env,
            catch_agent=catch_agent,
            episode_index=episode_index,
            max_steps_per_episode=max_steps_per_episode,
            success_distance_threshold=success_distance_threshold,
        )
        print(f"[Action:ground_grasp] success={_yes_no(ground_action_ok)}")
        if not ground_action_ok:
            continue

        step_decision, step_ok = check_decision("stepping")
        if not step_ok or step_decision != 1:
            continue
        step_result = _run_full_step(env, tai_agent, episode_index, max_steps_per_episode)
        step_action_ok = bool(step_result["success"])
        print(
            f"[Action:stepping] left={_yes_no(step_result['left_success'])}, "
            f"right={_yes_no(step_result['right_success'])}, success={_yes_no(step_action_ok)}"
        )
        if not step_action_ok:
            continue

        if hasattr(env, "straighten_legs_and_settle"):
            env.straighten_legs_and_settle(settle_ms=1800)
        _set_upper_phase_targets(1)
        upper_decision, upper_ok = check_decision("upper_grasp")
        if not upper_ok or upper_decision != 2:
            continue
        upper_left = _run_upper_left_grasp(
            env=env,
            upper_agent=upper_agent,
            cycle_idx=episode_index,
            stair_idx=2,
            max_steps=upper_max_steps,
        )
        upper_right_ok = False
        if upper_left["success"]:
            upper_right_ok = _replay_upper_right_grasp(
                env,
                upper_left["history"],
                left_hold_pose=upper_left.get("left_hold_pose"),
            )
        print(
            f"[Action:upper_grasp] left={_yes_no(upper_left['success'])}, "
            f"right={_yes_no(upper_right_ok)}, "
            f"success={_yes_no(upper_left['success'] and upper_right_ok)}"
        )

    correct = sum(values[0] for values in stage_stats.values())
    total = sum(values[1] for values in stage_stats.values())
    print("[Decision Summary]")
    for stage_name, (stage_correct, stage_total) in stage_stats.items():
        rate = stage_correct / float(stage_total) if stage_total else 0.0
        print(f"  - {stage_name}: {stage_correct}/{stage_total} = {rate:.4f}")
    overall_rate = correct / float(total) if total else 0.0
    print(f"  - overall: {correct}/{total} = {overall_rate:.4f}")
    return overall_rate


def run_batch_decision_models(
    robot,
    decision_dir: Optional[str] = None,
    catch_model_path: Optional[str] = None,
    tai_model_path: Optional[str] = None,
    upper_model_path: Optional[str] = None,
    episodes: int = 100,
    max_steps_per_episode: int = 20,
    upper_max_steps: int = 30,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
    save_json_path: Optional[str] = None,
):
    """Auto-load decision checkpoints and evaluate each model's decision success rate."""
    if decision_dir is None:
        decision_dir = path_list["model_path_decision_PPO_h"]

    decision_ckpts = _all_decision_ckpts(decision_dir)
    if not decision_ckpts:
        raise FileNotFoundError(f"no decision model checkpoint found in: {decision_dir}")

    env = Environment(robot)
    catch_agent = hppo(num_servos=2, state_dim=19)
    catch_model = _resolve_catch_model_path(catch_model_path, path_list["model_path_catch_PPO_h"])
    _load_catch_model(catch_agent, catch_model)
    print(f"loaded catch model for decision evaluation: {catch_model}")

    if tai_model_path is None:
        tai_model_path = _latest_ckpt(
            path_list["model_path_tai_PPO_h"],
            "tai_agent_*_*.ckpt",
            r"tai_agent_\d+_(\d+)\.ckpt$",
        )
    if not tai_model_path or not os.path.exists(tai_model_path):
        raise FileNotFoundError(f"stepping model not found: {tai_model_path}")
    tai_agent = hppo(num_servos=3, state_dim=28, use_image_input=False)
    _load_policy(tai_agent, tai_model_path, "policy_tai")
    print(f"loaded stepping model for decision evaluation: {tai_model_path}")

    if upper_model_path is None:
        upper_model_path = _latest_upper_ckpt(path_list["model_path_upper_PPO_h"])
    if not upper_model_path or not os.path.exists(upper_model_path):
        raise FileNotFoundError(f"upper grasp model not found: {upper_model_path}")
    upper_agent = hppo(
        num_servos=2,
        state_dim=15,
        use_image_input=True,
    )
    _load_policy(upper_agent, upper_model_path, "policy_upper")
    print(f"loaded upper grasp model for decision evaluation: {upper_model_path}")

    results = []
    for model_path in decision_ckpts:
        decision_agent = d_hppo(
            num_servos=1,
            state_dim=8,
            use_image_input=False,
            num_actions=3,
        )
        _load_decision_model(decision_agent, model_path)
        print("=" * 80)
        print(f"start evaluating decision model: {model_path}")

        success_rate = evaluate_one_decision_model(
            env=env,
            decision_agent=decision_agent,
            catch_agent=catch_agent,
            tai_agent=tai_agent,
            upper_agent=upper_agent,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            upper_max_steps=upper_max_steps,
            success_distance_threshold=success_distance_threshold,
            reset_wait_ms=reset_wait_ms,
        )

        result_row = {
            "model_path": model_path,
            "episodes": int(episodes),
            "success_rate": float(success_rate),
        }
        results.append(result_row)
        print(f"[decision-eval] model done: success_rate={success_rate:.4f}")

        # 每个模型完成后立即保存
        if save_json_path is None:
            test_log_dir = os.path.join(os.path.dirname(os.path.dirname(path_list["tai_log_path_PPO"])), "test_log")
            os.makedirs(test_log_dir, exist_ok=True)
            save_json_path = os.path.join(test_log_dir, "decision_model_success_rates.json")

        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存 {save_json_path}")

    print("=" * 80)
    print(f"all decision models evaluated, results saved to: {save_json_path}")
    return results


def decision_models_test(robot):
    """Direct-call interface: evaluate all decision models for 100 episodes each."""
    return run_batch_decision_models(
        robot=robot,
        episodes=100,
    )
