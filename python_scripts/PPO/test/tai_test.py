# -*- coding: utf-8 -*-
import glob
import json
import math
import os
import re
from typing import Optional

import numpy as np
import torch

from python_scripts.PPO.utils.Project_config import gps_goal, gps_goal1, path_list
from python_scripts.PPO.hppo.hppo_01 import HPPO as tai_hppo
from python_scripts.PPO.hppo.hppo_01 import HPPO as hppo
from python_scripts.PPO.utils.utils import with_ground_goal_state

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


def _all_tai_ckpts(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "tai_agent_*.ckpt"))
    if not files:
        return []

    def _num(file_path: str) -> int:
        match = re.search(r"tai_agent_\d+_(\d+)\.ckpt$", os.path.basename(file_path))
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


def _load_tai_model(agent, model_path: str) -> None:
    ckpt = _safe_torch_load(model_path)
    if isinstance(ckpt, dict):
        if "policy_tai" in ckpt:
            state_dict = ckpt["policy_tai"]
        elif "policy" in ckpt:
            state_dict = ckpt["policy"]
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


def _tai_distance_to_goal(env) -> float:
    try:
        gps_values = validate_and_clean_data(env.print_gps())
        foot_gps = gps_values[4] if len(gps_values) > 4 else [0.0, 0.0, 0.0]
        x = float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
        y = float(foot_gps[2]) if len(foot_gps) > 2 else 0.0
        
        if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
            goal_x = float(gps_goal1[0])
            goal_y = float(gps_goal1[1])
        else:
            goal_x, goal_y = 0.058, 0.023
        
        dx = goal_x - x
        dy = goal_y - y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float("inf")


def _tai_success(env, distance_threshold: float = 0.04) -> bool:
    return _tai_distance_to_goal(env) <= distance_threshold


def _catch_episode(env, catch_agent, episode_index: int, max_steps_per_episode: int, success_distance_threshold: float) -> bool:
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
        img_name = f"tai_catch_eval_{episode_index}_{steps}.png"

        _, done, _, _, _ = env.step(robot_state, masked_shoulder, masked_arm, steps, catch_flag)

        prev_shoulder_action = masked_shoulder
        prev_arm_action = masked_arm

        if _catch_success(env, success_distance_threshold):
            ok = True
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
    return ok


def _tai_episode(env, tai_agent, episode_index: int, max_steps_per_episode: int, success_distance_threshold: float) -> bool:
    imgs = []
    steps = 0
    ok = False

    # 初始化状态
    robot_state = env.get_robot_state()
    prev_discrete_actions = [1.0, 1.0, 1.0]
    prev_robot_state = None
    last_action_LegUpper = robot_state[12] if len(robot_state) > 12 else 0.0
    last_action_LegLower = robot_state[13] if len(robot_state) > 13 else 0.0
    last_action_Ankle = robot_state[14] if len(robot_state) > 14 else 0.0

    while steps < max_steps_per_episode:
        _, obs_tensor = env.get_img(steps)
        raw_robot_state = env.get_robot_state()
        robot_state = with_ground_goal_state(raw_robot_state)
        
        tai_state = _build_tai_feature_state(
            env=env,
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_action_LegUpper, last_action_LegLower, last_action_Ankle],
            prev_discrete_actions=prev_discrete_actions,
        )

        tai_dict = tai_agent.choose_action([obs_tensor, tai_state], greedy=True)

        tai_discrete_action = tai_dict["discrete_action"]
        tai_continuous_action = tai_dict["continuous_action"]

        d0 = int(tai_discrete_action[0])
        d1 = int(tai_discrete_action[1])
        d2 = int(tai_discrete_action[2])

        action_LegUpper = float(tai_continuous_action[0])
        action_LegLower = float(tai_continuous_action[1])
        action_Ankle = float(tai_continuous_action[2])

        prev_discrete_actions = [float(d0), float(d1), float(d2)]

        # 根据离散动作决定是使用新动作还是保持上一动作
        if d0 == 0:
            action_LegUpper_exec = last_action_LegUpper
        else:
            action_LegUpper_exec = action_LegUpper

        if d1 == 0:
            action_LegLower_exec = last_action_LegLower
        else:
            action_LegLower_exec = action_LegLower

        if d2 == 0:
            action_Ankle_exec = last_action_Ankle
        else:
            action_Ankle_exec = action_Ankle

        last_action_LegUpper = action_LegUpper_exec
        last_action_LegLower = action_LegLower_exec
        last_action_Ankle = action_Ankle_exec

        gps_values = validate_and_clean_data(env.print_gps())

        _, _, done, good, goal, _ = env.step2(robot_state, action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec, steps)

        if _tai_success(env, success_distance_threshold):
            ok = True
            env.wait(100)
            break

        steps += 1
        if done == 1:
            break

        prev_robot_state = np.array(robot_state, dtype=np.float32)

    _clear_runtime_cache(imgs)
    return ok


def _safe_touch(env, name: str) -> float:
    try:
        return float(env.darwin.get_touch_sensor_value(name))
    except Exception:
        return 0.0


def _build_tai_feature_state(
    env,
    robot_state,
    prev_robot_state,
    prev_exec_actions,
    prev_discrete_actions,
    feature_dim: int = 28,
):
    rs = np.array(validate_and_clean_data(robot_state), dtype=np.float32).reshape(-1)
    if prev_robot_state is None:
        prs = rs.copy()
    else:
        prs = np.array(validate_and_clean_data(prev_robot_state), dtype=np.float32).reshape(-1)
        if prs.shape[0] != rs.shape[0]:
            prs = rs.copy()

    def _idx(i: int) -> float:
        return float(rs[i]) if 0 <= i < rs.shape[0] else 0.0

    def _didx(i: int) -> float:
        if 0 <= i < rs.shape[0] and 0 <= i < prs.shape[0]:
            return float(rs[i] - prs[i])
        return 0.0

    gps_values = validate_and_clean_data(env.print_gps())
    foot_x = float(gps_values[4][1]) if len(gps_values) > 4 and len(gps_values[4]) > 1 else 0.0
    foot_y = float(gps_values[4][2]) if len(gps_values) > 4 and len(gps_values[4]) > 2 else 0.0

    # 从 Supervisor 写入的文件中读取机器人初始世界坐标（X=左右, Y=高度, Z=前后）
    try:
        with open('D:\\project_MultiAgent_hppo\\python_scripts\\robot_pos.txt', 'r') as f:
            pos_str = f.read().strip()
            bx, by, bz = pos_str.split(',')
            body_x, body_y, body_z = float(bx), float(by), float(bz)
    except Exception:
        body_x, body_y, body_z = 0.0, 0.0, 0.0

    if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
        goal_x = float(gps_goal1[0])
        goal_y = float(gps_goal1[1])
    else:
        goal_x, goal_y = 0.058, 0.023

    dx = goal_x - foot_x
    dy = goal_y - foot_y
    dist = math.sqrt(dx * dx + dy * dy)

    try:
        acc = env.darwin.accelerometer.getValues()
        acc_feat = [
            (float(acc[0]) - 520.0) / 100.0,
            (float(acc[1]) - 500.0) / 100.0,
            (float(acc[2]) - 640.0) / 120.0,
        ]
    except Exception:
        acc_feat = [0.0, 0.0, 0.0]

    try:
        gyro = env.darwin.gyro.getValues()
        gyro_feat = [
            (float(gyro[0]) - 510.0) / 20.0,
            (float(gyro[1]) - 510.0) / 20.0,
            (float(gyro[2]) - 510.0) / 20.0,
        ]
    except Exception:
        gyro_feat = [0.0, 0.0, 0.0]

    foot_touches = [
        _safe_touch(env, 'foot_L1'),
        _safe_touch(env, 'foot_L2'),
        _safe_touch(env, 'foot_L3'),
    ]

    feature = [
        _idx(10), _idx(11), _idx(12), _idx(13), _idx(14), _idx(15),
        _didx(10), _didx(11), _didx(12), _didx(13), _didx(14), _didx(15),
        foot_x, foot_y, goal_x, goal_y,
        body_x, body_y, body_z,
        *acc_feat,
        float(prev_exec_actions[0]), float(prev_exec_actions[1]), float(prev_exec_actions[2]),
        float(prev_discrete_actions[0]), float(prev_discrete_actions[1]), float(prev_discrete_actions[2]),
    ]

    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape[0] > int(feature_dim):
        feature = feature[:int(feature_dim)]
    elif feature.shape[0] < int(feature_dim):
        pad = np.zeros(int(feature_dim) - feature.shape[0], dtype=np.float32)
        feature = np.concatenate((feature, pad), axis=0)
    return feature


def evaluate_one_tai_model(
    env,
    tai_agent,
    catch_agent,
    episodes: int = 100,
    max_steps_per_catch: int = 20,
    max_steps_per_tai: int = 20,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
):
    success_count = 0
    valid_episode_count = 0
    attempt_count = 0

    env.reset()
    env.wait(reset_wait_ms)

    while valid_episode_count < episodes:
        # 每个尝试开始前确保环境清洁
        env.reset()
        env.wait(reset_wait_ms)
        
        attempt_count += 1
        print("\n==============================")
        print(f"🌍 Attempt {attempt_count} (Valid Episode {valid_episode_count + 1})")
        print("==============================")

        # 1) 先执行抓取
        catch_ok = _catch_episode(
            env=env,
            catch_agent=catch_agent,
            episode_index=attempt_count,
            max_steps_per_episode=max_steps_per_catch,
            success_distance_threshold=success_distance_threshold,
        )

        if not catch_ok:
            print("[Attempt Result]")
            print(f"  - 抓取成功: 否（此次尝试不计入有效周期）")
            print(f"  - 已完成有效周期: {valid_episode_count}/{episodes}")
            continue
        
        # 抓取成功，增加有效周期计数
        valid_episode_count += 1

        print("[Tai Test] catch succeeded, start step test")

        # 2) 抓取成功后，直接抬起腿进行踩踏
        env.darwin.tai_leg_L1()
        env.darwin.tai_leg_L2()
        env.wait(200)

        tai_ok = _tai_episode(
            env=env,
            tai_agent=tai_agent,
            episode_index=valid_episode_count,
            max_steps_per_episode=max_steps_per_tai,
            success_distance_threshold=success_distance_threshold,
        )

        # 只有两个都成功才算本周期成功
        ok = bool(catch_ok and tai_ok)
        success_count += int(ok)

        print("[Episode Result]")
        print("  - catch_success: True")
        print(f"  - tai_success: {bool(tai_ok)}")
        print(f"  - 本周期结果: {'成功' if ok else '失败'}")
        print(f"  - 当前累计成功率: {success_count}/{valid_episode_count} = {success_count / valid_episode_count:.4f}")

    return success_count / float(episodes) if episodes > 0 else 0.0


def run_batch_tai_models(
    robot,
    tai_dir: Optional[str] = None,
    catch_model_path: Optional[str] = None,
    episodes: int = 100,
    max_steps_per_catch: int = 20,
    max_steps_per_tai: int = 20,
    success_distance_threshold: float = 0.04,
    reset_wait_ms: int = 500,
    save_json_path: Optional[str] = None,
):
    if tai_dir is None:
        tai_dir = path_list["model_path_tai_PPO_h"]

    tai_ckpts = _all_tai_ckpts(tai_dir)
    if not tai_ckpts:
        raise FileNotFoundError(f"no tai model checkpoint found in: {tai_dir}")

    env = Environment(robot)
    catch_agent = hppo(num_servos=2, state_dim=19)
    catch_model = _resolve_catch_model_path(catch_model_path, path_list["model_path_catch_PPO_h"])
    _load_catch_model(catch_agent, catch_model)
    print(f"loaded catch model for tai evaluation: {catch_model}")

    results = []
    for model_path in tai_ckpts:
        tai_agent = tai_hppo(
            num_servos=3,
            state_dim=28,
            use_image_input=False,
        )
        _load_tai_model(tai_agent, model_path)
        print("=" * 80)
        print(f"start evaluating tai model: {model_path}")

        success_rate = evaluate_one_tai_model(
            env=env,
            tai_agent=tai_agent,
            catch_agent=catch_agent,
            episodes=episodes,
            max_steps_per_catch=max_steps_per_catch,
            max_steps_per_tai=max_steps_per_tai,
            success_distance_threshold=success_distance_threshold,
            reset_wait_ms=reset_wait_ms,
        )

        result_row = {
            "model_path": model_path,
            "episodes": int(episodes),
            "success_rate": float(success_rate),
        }
        results.append(result_row)
        print(f"[tai-eval] model done: success_rate={success_rate:.4f}")

        # 每个模型完成后立即保存
        if save_json_path is None:
            test_log_dir = os.path.join(os.path.dirname(os.path.dirname(path_list["tai_log_path_PPO"])), "test_log")
            os.makedirs(test_log_dir, exist_ok=True)
            save_json_path = os.path.join(test_log_dir, "tai_ground_success_rates.json")

        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存 {save_json_path}")

    print("=" * 80)
    print(f"all tai models evaluated, results saved to: {save_json_path}")
    return results


def tai_models_test(robot):
    """Direct-call interface: evaluate all tai models for 100 episodes each."""
    return run_batch_tai_models(
        robot=robot,
        episodes=100,
    )
