# -*- coding: utf-8 -*-
"""
Test suite for upper-stage stair grasp only.
Loads ground-grasp, stepping and upper-grasp action models.
"""

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch

from python_scripts.PPO.hppo.hppo_01 import HPPO as ActionHPPO
from python_scripts.PPO.test.full_climb_test import (
    _reset_env_and_wait,
    _run_full_step,
    _run_ground_both_grasp,
)
from python_scripts.PPO.utils.Project_config import Darwin_config, path_list
from python_scripts.PPO.utils.Webots_interfaces import Environment


STAIR_HEIGHT_M = 0.07
DEFAULT_MAX_CYCLES = 100
DEFAULT_MAX_STEPS = 20
UPPER_TARGET_X = 0.268
UPPER_TARGET_Y = 0.252
UPPER_GRASP_DISTANCE_THRESHOLD = 0.035
RIGHT_GRASP_DISTANCE_THRESHOLD = 0.05
GROUND_GRASP_DISTANCE_THRESHOLD = 0.04
UPPER_LEFT_FOOT_DISTANCE_THRESHOLD = 0.015
UPPER_WITHDRAW_DISTANCE_THRESHOLD = 0.05
ARM_DELTA_LIMIT = 0.3


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _safe_torch_load(ckpt_path: str):
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.load(ckpt_path, map_location=map_location)


def _latest_ckpt(dir_path: str, pattern: str, regex: str):
    files = glob.glob(os.path.join(dir_path, pattern))
    if not files:
        return None

    def _num(file_path: str) -> int:
        match = re.search(regex, os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return max(files, key=_num)


def _latest_upper_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "upper_agent_*_*.ckpt"))
    if not files:
        return None

    def _nums(file_path: str):
        match = re.search(r"upper_agent_(\d+)_(\d+)\.ckpt$", os.path.basename(file_path))
        if not match:
            return -1, -1
        return int(match.group(1)), int(match.group(2))

    return max(files, key=_nums)


def _all_upper_ckpts(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "upper_agent_*_*.ckpt"))

    def _nums(file_path: str):
        match = re.search(r"upper_agent_(\d+)_(\d+)\.ckpt$", os.path.basename(file_path))
        if not match:
            return -1, -1
        return int(match.group(1)), int(match.group(2))

    return sorted(files, key=_nums)


def _strip_common_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "policy.",
        "policy_upper.",
        "policy_tai.",
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


def _load_policy(agent, ckpt_path: str, policy_key: str):
    ckpt = _safe_torch_load(ckpt_path)
    if isinstance(ckpt, dict) and policy_key in ckpt:
        state_dict = ckpt[policy_key]
    elif isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        agent.policy.load_state_dict(state_dict)
        return

    try:
        agent.policy.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    normalized_state_dict = _normalize_state_dict(state_dict)
    if isinstance(ckpt, dict) and policy_key in ckpt:
        if normalized_state_dict != state_dict:
            try:
                agent.policy.load_state_dict(normalized_state_dict)
                return
            except RuntimeError:
                pass

    compatible_state_dict = _filter_compatible_state_dict(agent.policy, normalized_state_dict)
    if compatible_state_dict:
        agent.policy.load_state_dict(compatible_state_dict, strict=False)
        ignored = sorted(set(normalized_state_dict.keys()) - set(compatible_state_dict.keys()))
        print(
            f"[LoadPolicy] {os.path.basename(ckpt_path)} loaded with compatibility filtering; "
            f"ignored {len(ignored)} incompatible keys."
        )
        return

    agent.policy.load_state_dict(normalized_state_dict, strict=False)


def _safe_touch(env, name: str) -> float:
    try:
        return float(env.darwin.get_touch_sensor_value(name))
    except Exception:
        return 0.0


def _left_grasp_success(env) -> bool:
    return any(
        _safe_touch(env, name) > 0.0
        for name in ("grasp_L1", "grasp_L1_1", "grasp_L1_2")
    )


def _right_grasp_contact(env) -> bool:
    return any(
        _safe_touch(env, name) > 0.0
        for name in ("grasp_R1", "grasp_R1_1", "grasp_R1_2")
    )


def _right_grasp_distance_to_goal(env) -> float:
    try:
        gps_values = env.print_gps()
        right_gps = gps_values[1]
        right_x = float(right_gps[1]) if len(right_gps) > 1 else 0.0
        right_y = float(right_gps[2]) if len(right_gps) > 2 else 0.0
        dx = float(UPPER_TARGET_X) - right_x
        dy = float(UPPER_TARGET_Y) - right_y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        pass
    return float("inf")


def _right_grasp_success(env) -> bool:
    return _right_grasp_contact(env) and _right_grasp_distance_to_goal(env) < float(RIGHT_GRASP_DISTANCE_THRESHOLD)


def _open_left_gripper(env):
    if hasattr(env.darwin, "song_L"):
        env.darwin.song_L()
        return

    env.darwin.motors[20].setPosition(1.0)
    for _ in range(16):
        env.robot.step(env.timestep)


def _open_right_gripper(env):
    if hasattr(env.darwin, "song_R"):
        env.darwin.song_R()
        return

    env.darwin.motors[21].setPosition(1.0)
    for _ in range(16):
        env.robot.step(env.timestep)


def _hold_left_arm_pose(env, pose):
    if not pose:
        return
    for joint_idx, position in pose.items():
        env.darwin.motors[int(joint_idx)].setPosition(float(position))


def _upper_grasp_distance_to_goal(env) -> float:
    try:
        gps_values = env.print_gps()
        left_gps = gps_values[0]
        left_x = float(left_gps[1]) if len(left_gps) > 1 else 0.0
        left_y = float(left_gps[2]) if len(left_gps) > 2 else 0.0
        dx = float(UPPER_TARGET_X) - left_x
        dy = float(UPPER_TARGET_Y) - left_y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        pass
    return float("inf")


def _is_fallen(env) -> bool:
    try:
        if hasattr(env, "check_acceleration_and_gyro"):
            return not bool(env.check_acceleration_and_gyro())
        if hasattr(env.darwin, "check_acceleration_and_gyro"):
            return not bool(env.darwin.check_acceleration_and_gyro())
    except Exception:
        pass
    return False


def _prepare_upper_pose(env):
    metrics = {
        "left_foot_distance": None,
        "left_withdraw_distance": None,
    }
    if hasattr(env, "straighten_legs_and_settle"):
        env.straighten_legs_and_settle(settle_ms=1800)

    if hasattr(env, "left_foot_distance_to_goal"):
        left_foot_distance = float(env.left_foot_distance_to_goal())
        metrics["left_foot_distance"] = left_foot_distance
        print(f"[Upper Prepare] left_foot_distance={left_foot_distance:.4f}")
        if left_foot_distance > float(UPPER_LEFT_FOOT_DISTANCE_THRESHOLD):
            return False, "left_foot_distance", metrics

    if hasattr(env, "raise_head_for_upper_grasp"):
        env.raise_head_for_upper_grasp(settle_ms=600)
    if hasattr(env, "open_grippers_fully"):
        env.open_grippers_fully(settle_ms=800)
    if hasattr(env, "withdraw_left_from_step"):
        env.withdraw_left_from_step()
    if hasattr(env, "wait"):
        env.wait(3000)

    if hasattr(env, "catch_distance_to_goal"):
        left_withdraw_distance = float(env.catch_distance_to_goal())
        metrics["left_withdraw_distance"] = left_withdraw_distance
        print(f"[Upper Prepare] left_withdraw_distance={left_withdraw_distance:.4f}")
        if left_withdraw_distance <= float(UPPER_WITHDRAW_DISTANCE_THRESHOLD):
            return False, "left_withdraw_distance", metrics

    return True, None, metrics


def _run_upper_left_grasp(env, upper_agent, cycle_idx: int, stair_idx: int, max_steps: int):
    prepare_ok, prepare_failure, prepare_metrics = _prepare_upper_pose(env)
    if not prepare_ok:
        return {
            "success": False,
            "steps": 0,
            "done": 0,
            "goal": 0,
            "distance": _upper_grasp_distance_to_goal(env),
            "history": [],
            "prepare_failure": prepare_failure,
            "prepare_metrics": prepare_metrics,
        }

    _open_left_gripper(env)
    imgs = []
    prev_robot_state = None
    prev_discrete_actions = [1.0, 1.0]
    robot_state = env.get_robot_state()
    last_shoulder = float(robot_state[1]) if len(robot_state) > 1 else 0.0
    last_arm_upper = float(robot_state[3]) if len(robot_state) > 3 else 0.0
    last_arm_lower = float(robot_state[5]) if len(robot_state) > 5 else 0.0
    last_grasp = 0.0
    history = []
    done = 0
    goal = 0
    steps_executed = 0

    for steps in range(int(max_steps)):
        _, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()
        upper_state = env.build_upper_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_shoulder, last_arm_upper, last_arm_lower, last_grasp],
            prev_discrete_actions=prev_discrete_actions,
        )
        action_dict = upper_agent.choose_action([obs_tensor, upper_state], greedy=True)

        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]
        gate_shoulder = int(float(discrete_action[0]))
        gate_lower = int(float(discrete_action[1]))
        prev_discrete_actions = [float(gate_shoulder), float(gate_lower)]

        action_shoulder = float(continuous_action[0])
        action_lower = float(continuous_action[1])
        exec_shoulder = last_shoulder if gate_shoulder == 0 else action_shoulder
        exec_arm_upper = last_arm_upper
        exec_arm_lower = last_arm_lower if gate_lower == 0 else action_lower
        exec_grasp = 0.0
        last_shoulder = exec_shoulder
        last_arm_upper = exec_arm_upper
        last_arm_lower = exec_arm_lower
        last_grasp = exec_grasp

        current_left_shoulder = float(robot_state[1]) if len(robot_state) > 1 else 0.0
        current_right_shoulder = float(robot_state[0]) if len(robot_state) > 0 else 0.0
        current_left_lower = float(robot_state[5]) if len(robot_state) > 5 else 0.0
        current_right_lower = float(robot_state[4]) if len(robot_state) > 4 else 0.0
        left_shoulder_target = _clamp(
            0.2995 * float(exec_shoulder) - 0.145,
            Darwin_config.limit[1][0],
            Darwin_config.limit[1][1],
        )
        left_lower_target = _clamp(
            1.25 * float(exec_arm_lower) + 0.25,
            Darwin_config.limit[5][0],
            Darwin_config.limit[5][1],
        )
        delta_shoulder = _clamp(
            left_shoulder_target - current_left_shoulder,
            -ARM_DELTA_LIMIT,
            ARM_DELTA_LIMIT,
        )
        delta_lower = _clamp(
            left_lower_target - current_left_lower,
            -ARM_DELTA_LIMIT,
            ARM_DELTA_LIMIT,
        )

        _, _, done, _, goal, _ = env.step3(robot_state, exec_shoulder, exec_arm_lower, steps)
        env.wait(420)
        history.append(
            {
                "action": (exec_shoulder, exec_arm_lower),
                "left_target": (left_shoulder_target, left_lower_target),
                "left_start": (current_left_shoulder, current_left_lower),
                "right_start": (current_right_shoulder, current_right_lower),
                "delta": (delta_shoulder, delta_lower),
                "left_shoulder_delta": delta_shoulder,
                "left_arm_delta": delta_lower,
            }
        )
        steps_executed = steps + 1

        upper_distance = _upper_grasp_distance_to_goal(env)
        left_contact = _left_grasp_success(env)
        left_success = left_contact and upper_distance <= float(UPPER_GRASP_DISTANCE_THRESHOLD)
        print(
            f"  [Upper L {stair_idx}.{steps}] gate=({gate_shoulder},{gate_lower}), "
            f"left={left_success}, contact={left_contact}, dist={upper_distance:.4f}, goal={int(goal)}"
        )
        if left_success:
            hold_state = env.get_robot_state()
            left_hold_pose = {
                1: float(hold_state[1]) if len(hold_state) > 1 else 0.0,
                3: float(hold_state[3]) if len(hold_state) > 3 else 0.0,
                5: float(hold_state[5]) if len(hold_state) > 5 else 0.0,
            }
            env.darwin.motors[20].setPosition(-0.5)
            for _ in range(20):
                _hold_left_arm_pose(env, left_hold_pose)
                env.robot.step(env.timestep)
            return {
                "success": _left_grasp_success(env),
                "steps": steps_executed,
                "done": int(done),
                "goal": int(goal),
                "distance": upper_distance,
                "history": history,
                "left_hold_pose": left_hold_pose,
            }
        if done == 1:
            print(f"  [Upper L {stair_idx}.{steps}] env done before left grasp success")
            break
        prev_robot_state = np.array(robot_state, dtype=np.float32)

    return {
        "success": False,
        "steps": steps_executed,
        "done": int(done),
        "goal": int(goal),
        "distance": _upper_grasp_distance_to_goal(env),
        "history": history,
    }


def _wait_right_upper_targets(env, targets, left_hold_pose=None, timeout_steps: int = 120, tolerance: float = 0.012):
    wait_fn = getattr(getattr(env, "darwin", None), "wait_joint_targets_reached", None)
    if callable(wait_fn):
        return wait_fn(
            targets=targets,
            timeout_steps=timeout_steps,
            tolerance=tolerance,
            stable_need=2,
            post_settle_steps=6,
            success_check=lambda: _right_grasp_success(env),
        )

    for _ in range(int(timeout_steps)):
        _hold_left_arm_pose(env, left_hold_pose)
        env.robot.step(env.timestep)
        if _right_grasp_success(env):
            return True
    return False


def _run_upper_right_grasp(env, left_action_history, cycle_idx: int, stair_idx: int, max_steps: int, left_hold_pose=None, settle_steps: int = 20):
    _open_right_gripper(env)
    if hasattr(env, "withdraw_right_from_step"):
        env.withdraw_right_from_step()
    elif hasattr(env.darwin, "withdraw_right_from_step"):
        env.darwin.withdraw_right_from_step()
    if hasattr(env, "wait"):
        env.wait(600)
    _hold_left_arm_pose(env, left_hold_pose)

    history = []
    steps_executed = 0

    for steps, left_item in enumerate(list(left_action_history)[: int(max_steps)]):
        if isinstance(left_item, dict):
            left_action = left_item.get("action", (0.0, 0.0))
            left_shoulder_delta = float(left_item.get("left_shoulder_delta", left_item.get("delta", (0.0, 0.0))[0]))
            left_arm_delta = float(left_item.get("left_arm_delta", left_item.get("delta", (0.0, 0.0))[1]))
        else:
            left_action = left_item
            left_shoulder_delta = 0.0
            left_arm_delta = 0.0

        exec_shoulder = float(left_action[0])
        exec_arm_lower = float(left_action[1])
        robot_state = env.get_robot_state()
        current_right_shoulder = float(robot_state[0]) if len(robot_state) > 0 else 0.0
        current_right_lower = float(robot_state[4]) if len(robot_state) > 4 else 0.0

        right_shoulder_target = _clamp(
            current_right_shoulder - left_shoulder_delta,
            Darwin_config.limit[0][0],
            Darwin_config.limit[0][1],
        )
        right_lower_target = _clamp(
            current_right_lower - left_arm_delta,
            Darwin_config.limit[4][0],
            Darwin_config.limit[4][1],
        )
        shoulder_delta = right_shoulder_target - current_right_shoulder
        lower_delta = right_lower_target - current_right_lower

        env.darwin.motors[0].setPosition(right_shoulder_target)
        env.darwin.motors[4].setPosition(right_lower_target)
        reached = _wait_right_upper_targets(
            env,
            targets={0: right_shoulder_target, 4: right_lower_target},
            left_hold_pose=left_hold_pose,
        )
        _hold_left_arm_pose(env, left_hold_pose)

        history.append(
            {
                "action": (exec_shoulder, exec_arm_lower),
                "left_delta": (left_shoulder_delta, left_arm_delta),
                "right_start": (current_right_shoulder, current_right_lower),
                "right_target": (right_shoulder_target, right_lower_target),
                "delta": (shoulder_delta, lower_delta),
                "reached": bool(reached),
                "source": "left_delta_replay",
            }
        )
        steps_executed = steps + 1
        right_contact = _right_grasp_contact(env)
        right_distance = _right_grasp_distance_to_goal(env)
        right_success = right_contact and right_distance < float(RIGHT_GRASP_DISTANCE_THRESHOLD)
        print(
            f"  [Upper R Replay {stair_idx}.{steps}] "
            f"left_delta=({left_shoulder_delta:.4f},{left_arm_delta:.4f}), "
            f"right_start=({current_right_shoulder:.4f},{current_right_lower:.4f}), "
            f"right_targets ShoulderR={right_shoulder_target:.4f}, ArmLowerR={right_lower_target:.4f}, "
            f"reached={bool(reached)}, contact={right_contact}, dist={right_distance:.4f}, success={right_success}"
        )
        if right_success:
            env.darwin.motors[21].setPosition(-0.5)
            for _ in range(int(settle_steps)):
                _hold_left_arm_pose(env, left_hold_pose)
                env.robot.step(env.timestep)
            return {
                "success": _right_grasp_success(env),
                "steps": steps_executed,
                "history": history,
            }

    env.darwin.motors[21].setPosition(-0.5)
    for _ in range(int(settle_steps)):
        _hold_left_arm_pose(env, left_hold_pose)
        env.robot.step(env.timestep)
        if _right_grasp_success(env):
            return {
                "success": True,
                "steps": steps_executed,
                "history": history,
            }
    return {
        "success": False,
        "steps": steps_executed,
        "history": history,
    }

def _load_agents(catch_model_dir, tai_model_dir, upper_model_dir):
    catch_ckpt = _latest_ckpt(catch_model_dir, "catch_hppo_*.ckpt", r"catch_hppo_(\d+)\.ckpt$")
    tai_ckpt = _latest_ckpt(tai_model_dir, "tai_agent_*_*.ckpt", r"tai_agent_\d+_(\d+)\.ckpt$")
    upper_ckpt = _latest_upper_ckpt(upper_model_dir)

    missing = {
        "catch": catch_ckpt,
        "tai": tai_ckpt,
        "upper": upper_ckpt,
    }
    if any(path is None for path in missing.values()):
        raise FileNotFoundError(
            "Missing PPO checkpoint(s): "
            + ", ".join(f"{name}={path}" for name, path in missing.items())
        )

    catch_agent = ActionHPPO(num_servos=2, state_dim=19)
    tai_agent = ActionHPPO(
        num_servos=3,
        state_dim=28,
        use_image_input=False,
    )
    upper_agent = ActionHPPO(
        num_servos=2,
        state_dim=15,
        use_image_input=True,
    )
    _load_policy(catch_agent, catch_ckpt, "policy")
    _load_policy(tai_agent, tai_ckpt, "policy_tai")
    _load_policy(upper_agent, upper_ckpt, "policy_upper")
    print(f"Loaded catch policy: {catch_ckpt}")
    print(f"Loaded tai policy: {tai_ckpt}")
    print(f"Loaded upper left policy: {upper_ckpt}")
    return catch_agent, tai_agent, upper_agent, missing


def _run_one_upper_grasp_cycle(
    env,
    catch_agent,
    tai_agent,
    upper_agent,
    cycle_idx: int,
    max_steps: int,
):
    _reset_env_and_wait(env)

    ground_result = _run_ground_both_grasp(env, catch_agent, cycle_idx, max_steps)
    if not ground_result["success"]:
        return {"success": False, "counted": False, "failure_stage": "ground_grasp"}

    ground_distance = float(env.catch_distance_to_goal()) if hasattr(env, "catch_distance_to_goal") else float("inf")
    print(f"[Cycle {cycle_idx:03d}] ground grasp distance_to_goal={ground_distance:.4f}")
    if ground_distance > float(GROUND_GRASP_DISTANCE_THRESHOLD):
        print(f"[Cycle {cycle_idx:03d}] ground grasp hit wrong stair, reset and retry.")
        return {"success": False, "counted": False, "failure_stage": "ground_grasp_wrong_target"}

    first_step = _run_full_step(env, tai_agent, cycle_idx, max_steps)
    if not first_step["success"]:
        return {"success": False, "counted": False, "failure_stage": "first_step"}

    upper_left = _run_upper_left_grasp(env, upper_agent, cycle_idx, 2, max_steps)
    if not upper_left["success"]:
        failure_stage = upper_left.get("prepare_failure") or "upper_left_grasp"
        print(
            f"[Cycle {cycle_idx:03d}] failed at {failure_stage}, "
            f"left_steps={int(upper_left.get('steps', 0))}, "
            f"left_distance={float(upper_left.get('distance', float('inf'))):.4f}"
        )
        return {
            "success": False,
            "counted": upper_left.get("prepare_failure") is None,
            "failure_stage": failure_stage,
            "upper_left_steps": int(upper_left.get("steps", 0)),
            "upper_left_distance": float(upper_left.get("distance", float("inf"))),
        }

    upper_right = _run_upper_right_grasp(
        env,
        upper_left.get("history", []),
        cycle_idx,
        2,
        max_steps,
        left_hold_pose=upper_left.get("left_hold_pose"),
    )
    if not upper_right["success"]:
        print(f"[Cycle {cycle_idx:03d}] failed at upper_right_grasp")
        return {
            "success": False,
            "counted": True,
            "failure_stage": "upper_right_grasp",
            "upper_left_steps": int(upper_left.get("steps", 0)),
            "upper_left_distance": float(upper_left.get("distance", float("inf"))),
            "upper_right_steps": int(upper_right.get("steps", 0)),
        }

    print(f"[Cycle {cycle_idx:03d}] upper grasp success")
    return {
        "success": True,
        "counted": True,
        "failure_stage": None,
        "upper_left_steps": int(upper_left.get("steps", 0)),
        "upper_left_distance": float(upper_left.get("distance", float("inf"))),
        "upper_right_steps": int(upper_right.get("steps", 0)),
    }


def upper_grasp_test(
    robot,
    catch_model_dir=None,
    tai_model_dir=None,
    upper_model_dir=None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_stairs_per_cycle=None,
    result_file=None,
):
    catch_model_dir = catch_model_dir or path_list["model_path_catch_PPO_h"]
    tai_model_dir = tai_model_dir or path_list["model_path_tai_PPO_h"]
    upper_model_dir = upper_model_dir or path_list["model_path_upper_PPO_h"]
    test_log_dir = os.path.join(os.path.dirname(os.path.dirname(path_list["tai_log_path_PPO"])), "test_log")
    result_file = result_file or os.path.join(
        test_log_dir,
        "upper_grasp_model_success_rates.json",
    )
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    upper_ckpts = _all_upper_ckpts(upper_model_dir)
    if not upper_ckpts:
        raise FileNotFoundError(f"Missing upper checkpoint(s): {upper_model_dir}")

    catch_agent, tai_agent, upper_agent, _ = _load_agents(
        catch_model_dir,
        tai_model_dir,
        upper_model_dir,
    )
    env = Environment(robot)
    results = []

    def _save_results():
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)

    for model_idx, upper_ckpt in enumerate(upper_ckpts, 1):
        print(f"\n========== Upper Model {model_idx:03d}/{len(upper_ckpts)}: {os.path.basename(upper_ckpt)} ==========")
        _load_policy(upper_agent, upper_ckpt, "policy_upper")
        success_count = 0
        valid_cycle_count = 0
        attempt_count = 0

        while valid_cycle_count < int(max_cycles):
            attempt_count += 1
            print("\n==============================")
            print(f"Attempt {attempt_count} (Valid Episode {valid_cycle_count + 1})")
            print("==============================")

            result = _run_one_upper_grasp_cycle(
                env,
                catch_agent,
                tai_agent,
                upper_agent,
                attempt_count,
                int(max_steps),
            )
            if not result.get("counted", False):
                print("[Attempt Result]")
                print(f"  - pre_flow_success: False (failure_stage: {result.get('failure_stage')})")
                print(f"  - completed_valid_cycles: {valid_cycle_count}/{max_cycles}")
                continue

            valid_cycle_count += 1
            if result["success"]:
                success_count += 1

            print("[Episode Result]")
            print("  - pre_flow_success: True")
            print(f"  - upper_left_success: {result.get('failure_stage') != 'upper_left_grasp'}")
            print(f"  - upper_right_success: {bool(result.get('success'))}")
            print(f"  - episode_result: {'success' if result.get('success') else 'failure'}")
            print(f"  - current_success_rate: {success_count}/{valid_cycle_count} = {success_count / valid_cycle_count:.4f}")

        rate = success_count / float(valid_cycle_count) if valid_cycle_count > 0 else 0.0
        model_result = {
            "model_path": upper_ckpt,
            "episodes": int(valid_cycle_count),
            "success_rate": float(rate),
        }
        results.append(model_result)
        _save_results()
        print(f"[Upper] {os.path.basename(upper_ckpt)} => {success_count} / {valid_cycle_count} ({rate * 100:.2f}%)")
        print(f"[Upper] snapshot saved to: {result_file}")

    print("=" * 70)
    print(f"Upper grasp only test done. Result saved to: {result_file}")
    print("Upper model success rates:")
    for item in sorted(results, key=lambda value: value["success_rate"], reverse=True):
        print(f"  {os.path.basename(item['model_path'])}: {item['success_rate'] * 100:.2f}%")
    print("=" * 70)
    return results


SCRIPT_CONFIG = {
    "catch_dir": None,
    "tai_dir": None,
    "upper_dir": None,
    "max_cycles": DEFAULT_MAX_CYCLES,
    "max_steps": DEFAULT_MAX_STEPS,
    "result_file": None,
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run upper-stage grasp only test.")
    parser.add_argument("--catch-dir", default=SCRIPT_CONFIG["catch_dir"])
    parser.add_argument("--tai-dir", default=SCRIPT_CONFIG["tai_dir"])
    parser.add_argument("--upper-dir", default=SCRIPT_CONFIG["upper_dir"])
    parser.add_argument("--max-cycles", type=int, default=SCRIPT_CONFIG["max_cycles"])
    parser.add_argument("--max-steps", type=int, default=SCRIPT_CONFIG["max_steps"])
    parser.add_argument("--result-file", default=SCRIPT_CONFIG["result_file"])
    return parser.parse_args()


def main():
    controller_mod = __import__("controller")
    robot = controller_mod.Robot()
    args = _parse_args()
    upper_grasp_test(
        robot=robot,
        catch_model_dir=args.catch_dir,
        tai_model_dir=args.tai_dir,
        upper_model_dir=args.upper_dir,
        max_cycles=args.max_cycles,
        result_file=args.result_file,
    )


if __name__ == "__main__":
    main()






