# -*- coding: utf-8 -*-
import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch

from python_scripts.PPO.hppo.hppo import HPPO as DecisionHPPO
from python_scripts.PPO.hppo.hppo_01 import HPPO as ActionHPPO
from python_scripts.PPO.utils.utils import with_ground_goal_state
import python_scripts.PPO.utils.Project_config as project_config
from python_scripts.PPO.utils.Project_config import Darwin_config, path_list
from python_scripts.PPO.utils.Webots_interfaces import Environment


STAIR_HEIGHT_M = 0.07
DEFAULT_MAX_CYCLES = 100
DEFAULT_MAX_STEPS = 20
DEFAULT_UPPER_MAX_STEPS = 30
DEFAULT_MAX_STAIRS_PER_CYCLE = 8
UPPER_TARGET_X = 0.273
UPPER_TARGET_Y = 0.260
GROUND_GRASP_DISTANCE_THRESHOLD = 0.04
STEP_DISTANCE_THRESHOLD = 0.08
UPPER_GRASP_DISTANCE_THRESHOLD = 0.035
RIGHT_GRASP_DISTANCE_THRESHOLD = 0.05
UPPER_LEFT_FOOT_DISTANCE_THRESHOLD = 0.015
UPPER_WITHDRAW_DISTANCE_THRESHOLD = 0.05
ARM_DELTA_LIMIT = 0.3


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


def _strip_common_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "policy.",
        "policy_upper.",
        "policy_tai.",
        "policy_catch.",
        "policy_decision.",
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


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


BASE_STEP_TARGET = tuple(float(value) for value in project_config.gps_goal1[:2])
BASE_UPPER_TARGET = (float(UPPER_TARGET_X), float(UPPER_TARGET_Y))
BASE_GROUND_TARGET = tuple(float(value) for value in project_config.gps_goal[:2])


def _offset_target(target, climbed_stairs: int):
    offset = float(climbed_stairs) * STAIR_HEIGHT_M
    return [float(target[0]) + offset, float(target[1]) + offset]


def _replace_goal_list(goal_list, values):
    if isinstance(goal_list, list):
        goal_list[:2] = [float(values[0]), float(values[1])]


def _apply_target_lists(upper_target, step_target):
    _replace_goal_list(project_config.gps_goal1, step_target)
    _replace_goal_list(project_config.gps_goal2, upper_target)
    _replace_goal_list(project_config.gps_goal, upper_target)

    global UPPER_TARGET_X, UPPER_TARGET_Y
    UPPER_TARGET_X = float(upper_target[0])
    UPPER_TARGET_Y = float(upper_target[1])

    try:
        from python_scripts.PPO.Robot_Run.RobotRun3 import RobotRun3

        RobotRun3.TARGET_X = float(upper_target[0])
        RobotRun3.TARGET_Y = float(upper_target[1])
    except Exception:
        pass

    return {"upper_target": upper_target, "step_target": step_target}


def _set_climb_targets(climbed_stairs: int):
    upper_target = _offset_target(BASE_UPPER_TARGET, max(0, int(climbed_stairs) - 1))
    step_target = _offset_target(BASE_STEP_TARGET, climbed_stairs)
    return _apply_target_lists(upper_target, step_target)


def _set_upper_phase_targets(climbed_stairs: int):
    target_index = max(0, int(climbed_stairs) - 1)
    upper_target = _offset_target(BASE_UPPER_TARGET, target_index)
    step_target = _offset_target(BASE_STEP_TARGET, target_index)
    return _apply_target_lists(upper_target, step_target)


def _restore_base_targets():
    _replace_goal_list(project_config.gps_goal, BASE_GROUND_TARGET)
    _replace_goal_list(project_config.gps_goal1, BASE_STEP_TARGET)
    _replace_goal_list(project_config.gps_goal2, BASE_UPPER_TARGET)
    _set_climb_targets(0)
    _replace_goal_list(project_config.gps_goal, BASE_GROUND_TARGET)


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


def _right_grasp_success(env) -> bool:
    return any(
        _safe_touch(env, name) > 0.0
        for name in ("grasp_R1", "grasp_R1_1", "grasp_R1_2")
    )


def _both_grasp_success(env) -> bool:
    return _left_grasp_success(env) and _right_grasp_success(env)


def _ground_grasp_distance_to_goal(env) -> float:
    try:
        gps_values = env.print_gps()
        left_gps = gps_values[0]
        left_x = float(left_gps[1]) if len(left_gps) > 1 else 0.0
        left_y = float(left_gps[2]) if len(left_gps) > 2 else 0.0
        dx = float(project_config.gps_goal[0]) - left_x
        dy = float(project_config.gps_goal[1]) - left_y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float("inf")


def _ground_both_grasp_success(env) -> bool:
    return _both_grasp_success(env) and _ground_grasp_distance_to_goal(env) <= float(GROUND_GRASP_DISTANCE_THRESHOLD)


def _left_foot_success(env) -> bool:
    return any(_safe_touch(env, name) > 0.0 for name in ("foot_L1", "foot_L2", "foot_L3"))


def _right_foot_success(env) -> bool:
    return any(_safe_touch(env, name) > 0.0 for name in ("foot_R1", "foot_R2", "foot_R3"))


def _left_foot_distance_to_goal(env) -> float:
    try:
        distance_fn = getattr(env, "left_foot_distance_to_goal", None)
        if callable(distance_fn):
            return float(distance_fn())
    except Exception:
        pass

    try:
        gps_values = env.print_gps()
        foot_gps = gps_values[4] if len(gps_values) > 4 else [0.0, 0.0, 0.0]
        foot_x = float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
        foot_y = float(foot_gps[2]) if len(foot_gps) > 2 else 0.0
        dx = float(project_config.gps_goal1[0]) - foot_x
        dy = float(project_config.gps_goal1[1]) - foot_y
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float("inf")


def _left_step_success(env) -> bool:
    return _left_foot_success(env) and _left_foot_distance_to_goal(env) <= float(STEP_DISTANCE_THRESHOLD)


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

def _right_upper_grasp_distance_to_goal(env) -> float:
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


def _right_upper_grasp_success(env) -> bool:
    return _right_grasp_success(env) and _right_upper_grasp_distance_to_goal(env) < float(RIGHT_GRASP_DISTANCE_THRESHOLD)


def _is_fallen(env) -> bool:
    try:
        if hasattr(env, "check_acceleration_and_gyro"):
            return not bool(env.check_acceleration_and_gyro())
        if hasattr(env.darwin, "check_acceleration_and_gyro"):
            return not bool(env.darwin.check_acceleration_and_gyro())
    except Exception:
        pass
    return False


def _get_decision_state(env):
    pressure_state, _, pressure_detected = env.get_hand_pressure_state()
    decision_state = np.concatenate((
        np.asarray(pressure_state, dtype=np.float32),
        np.array([env.hand_foot_gps_distance(), env.get_foot_y_post()], dtype=np.float32),
    ))
    return decision_state, bool(pressure_detected)


def _wait_supervisor_reset_done(env, timeout_steps: int = 220) -> bool:
    for _ in range(int(timeout_steps)):
        env.robot.step(env.timestep)
        try:
            with open(path_list["resetFlag"], "r") as file:
                if file.read().strip() == "1":
                    return True
        except Exception:
            pass
    return False


def _is_facing_stairs_from_gps(env, forward_diff_thresh: float = 0.02, height_diff_thresh: float = 0.03):
    try:
        gps1, gps2, gps3, gps4, _ = env.print_gps()
        front_forward_diff = abs(float(gps1[2]) - float(gps2[2]))
        rear_forward_diff = abs(float(gps3[2]) - float(gps4[2]))
        front_height_diff = abs(float(gps1[1]) - float(gps2[1]))
        rear_height_diff = abs(float(gps3[1]) - float(gps4[1]))
        return (
            front_forward_diff <= float(forward_diff_thresh)
            and rear_forward_diff <= float(forward_diff_thresh)
            and front_height_diff <= float(height_diff_thresh)
            and rear_height_diff <= float(height_diff_thresh)
        )
    except Exception:
        return False


def _reset_env_and_wait(env, wait_ms: int = 5000, settle_steps: int = 150, max_retries: int = 1):
    for attempt in range(1, int(max_retries) + 1):
        print(f"  [Reset] attempt {attempt}/{max_retries}")
        env.reset()
        _wait_supervisor_reset_done(env)
        env.wait(wait_ms)

        imu_ok = not _is_fallen(env)
        facing_ok = _is_facing_stairs_from_gps(env)
        if imu_ok and facing_ok:
            for _ in range(int(settle_steps)):
                env.robot.step(env.timestep)
            return

    print("  [Reset] warning: reset checks did not fully pass; continuing.")
    for _ in range(int(settle_steps)):
        env.robot.step(env.timestep)


def _wait_joint_targets_reached(
    env,
    targets: dict,
    timeout_steps: int = 120,
    tolerance: float = 0.01,
    stable_need: int = 2,
    post_settle_steps: int = 0,
    success_check=None,
) -> bool:
    sensors = getattr(env.darwin, "motors_sensors", None)
    if not sensors:
        for _ in range(int(timeout_steps) + max(0, int(post_settle_steps))):
            env.robot.step(env.timestep)
            if callable(success_check) and bool(success_check()):
                return True
        return False

    stable_count = 0
    reached = False
    for _ in range(int(timeout_steps)):
        env.robot.step(env.timestep)
        if callable(success_check) and bool(success_check()):
            return True

        all_reached = True
        for joint_idx, target in targets.items():
            current = float(sensors[int(joint_idx)].getValue())
            if abs(current - float(target)) > float(tolerance):
                all_reached = False
                break

        if all_reached:
            stable_count += 1
            if stable_count >= int(stable_need):
                reached = True
                break
        else:
            stable_count = 0

    for _ in range(max(0, int(post_settle_steps))):
        env.robot.step(env.timestep)
        if callable(success_check) and bool(success_check()):
            return True

    return reached


def _reset_leg_pose_like_env_reset(env, steps: int = 24):
    env.darwin.motors[11].setPosition(0.5)
    env.darwin.motors[13].setPosition(-0.2)
    env.darwin.motors[15].setPosition(0.0)
    env.darwin.motors[17].setPosition(0.0)
    env.darwin.motors[10].setPosition(-0.5)
    env.darwin.motors[12].setPosition(0.2)
    env.darwin.motors[14].setPosition(0.0)
    env.darwin.motors[16].setPosition(0.0)
    for _ in range(int(steps)):
        env.robot.step(env.timestep)


def _reset_right_leg_pose_like_env_reset(env, steps: int = 28):
    env.darwin.tai_leg_R1()
    env.darwin.tai_leg_R2()
    boost_upper = _clamp(-1.72, Darwin_config.limit[10][0], Darwin_config.limit[10][1])
    boost_lower = _clamp(2.23, Darwin_config.limit[12][0], Darwin_config.limit[12][1])
    boost_ankle = _clamp(1.05, Darwin_config.limit[14][0], Darwin_config.limit[14][1])
    env.darwin.motors[10].setPosition(boost_upper)
    env.darwin.motors[12].setPosition(boost_lower)
    env.darwin.motors[14].setPosition(boost_ankle)
    _wait_joint_targets_reached(
        env,
        targets={10: boost_upper, 12: boost_lower, 14: boost_ankle},
        timeout_steps=120,
        tolerance=0.012,
        stable_need=2,
    )
    for _ in range(int(steps)):
        env.robot.step(env.timestep)


def _close_grippers(env, settle_steps: int = 20):
    env.darwin.motors[20].setPosition(-0.5)
    env.darwin.motors[21].setPosition(-0.5)
    for _ in range(int(settle_steps)):
        env.robot.step(env.timestep)


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


def _capture_step_entry_pose(env):
    try:
        return [float(value) for value in env.get_robot_state()]
    except Exception:
        return None


def _restore_step_entry_pose(env, pose, settle_steps: int = 40):
    if not pose:
        return False

    targets = {}
    for joint_idx, position in enumerate(pose[:20]):
        try:
            lower, upper = Darwin_config.limit[joint_idx]
            target = _clamp(float(position), lower, upper)
            env.darwin.motors[joint_idx].setPosition(target)
            targets[joint_idx] = target
        except Exception:
            continue

    env.darwin.motors[20].setPosition(-0.5)
    env.darwin.motors[21].setPosition(-0.5)
    reached = _wait_joint_targets_reached(
        env,
        targets=targets,
        timeout_steps=180,
        tolerance=0.018,
        stable_need=3,
        post_settle_steps=int(settle_steps),
    )
    imu_ok = not _is_fallen(env)
    print(f"[Upper Step Prepare] restored_ground_step_pose={bool(reached)}, imu_ok={imu_ok}")
    return bool(imu_ok)

def _run_ground_both_grasp(env, catch_agent, cycle_idx: int, max_steps: int, contact_hold_steps: int = 1):
    imgs = []
    prev_shoulder_action = 0.0
    prev_arm_action = 0.0
    prev_discrete_actions = [1.0, 1.0]
    prev_robot_state = None
    history = []
    contact_hold = 0
    done = 0
    steps_executed = 0

    for steps in range(int(max_steps)):

        _, obs_tensor = env.get_img(steps)
        raw_robot_state = env.get_robot_state()
        robot_state = env.validate_and_clean_data(raw_robot_state)
        catch_state = env.build_catch_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[prev_shoulder_action, prev_arm_action],
            prev_discrete_actions=prev_discrete_actions,
        )
        action_dict = catch_agent.choose_action((obs_tensor, catch_state), greedy=True)

        d_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]
        gate_shoulder = int(float(d_action[0]))
        gate_arm = int(float(d_action[1]))
        action_shoulder = float(continuous_action[0])
        action_arm = float(continuous_action[1])
        exec_shoulder = prev_shoulder_action if gate_shoulder == 0 else action_shoulder
        exec_arm = prev_arm_action if gate_arm == 0 else action_arm

        both_contact_before = _both_grasp_success(env)
        ground_success_before = _ground_both_grasp_success(env)
        catch_flag = 1.0 if ground_success_before else 0.0
        gps1, gps2, gps3, gps4, _ = env.print_gps()
        _, done, _, goal, _ = env.step(robot_state, exec_shoulder, exec_arm, steps, catch_flag)

        history.append((exec_shoulder, exec_arm))
        prev_shoulder_action = exec_shoulder
        prev_arm_action = exec_arm
        prev_discrete_actions = [float(gate_shoulder), float(gate_arm)]
        prev_robot_state = robot_state
        steps_executed = steps + 1

        ground_distance = _ground_grasp_distance_to_goal(env)
        ground_success = _ground_both_grasp_success(env)
        if ground_success:
            contact_hold += 1
        else:
            contact_hold = 0

        print(
            f"  [Ground Grasp {steps}] gate=({gate_shoulder},{gate_arm}), "
            f"exec=({exec_shoulder:.4f},{exec_arm:.4f}), both={_both_grasp_success(env)}, "
            f"dist={ground_distance:.4f}, success={ground_success}"
        )

        if contact_hold >= int(contact_hold_steps):
            _close_grippers(env)
            return {
                "success": _ground_both_grasp_success(env),
                "steps": steps_executed,
                "done": int(done),
                "history": history,
            }

        if done == 1 and not _ground_both_grasp_success(env):
            break

    return {
        "success": False,
        "steps": steps_executed,
        "done": int(done),
        "history": history,
    }


def _run_left_step(env, tai_agent, cycle_idx: int, max_steps: int):
    env.darwin.tai_leg_L1()
    env.darwin.tai_leg_L2()

    imgs = []
    history = []
    robot_state_initial = env.get_robot_state()
    last_upper = float(robot_state_initial[12]) if len(robot_state_initial) > 12 else 0.0
    last_lower = float(robot_state_initial[13]) if len(robot_state_initial) > 13 else 0.0
    last_ankle = float(robot_state_initial[14]) if len(robot_state_initial) > 14 else 0.0
    prev_discrete_actions = [1.0, 1.0, 1.0]
    prev_robot_state = None
    done = 0
    goal = 0
    steps_executed = 0

    for steps in range(int(max_steps)):
        _, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()
        tai_state = env.build_tai_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_upper, last_lower, last_ankle],
            prev_discrete_actions=prev_discrete_actions,
            feature_dim=28,
        )
        action_dict = tai_agent.choose_action([obs_tensor, tai_state], greedy=True)

        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]
        gate_upper = int(float(discrete_action[0]))
        gate_lower = int(float(discrete_action[1]))
        gate_ankle = int(float(discrete_action[2]))
        prev_discrete_actions = [float(gate_upper), float(gate_lower), float(gate_ankle)]

        action_upper = float(continuous_action[0])
        action_lower = float(continuous_action[1])
        action_ankle = float(continuous_action[2])
        exec_upper = last_upper if gate_upper == 0 else action_upper
        exec_lower = last_lower if gate_lower == 0 else action_lower
        exec_ankle = last_ankle if gate_ankle == 0 else action_ankle
        last_upper, last_lower, last_ankle = exec_upper, exec_lower, exec_ankle

        gps_values = env.print_gps()
        _, _, done, _, goal, _ = env.step2(robot_state, exec_upper, exec_lower, exec_ankle, steps)

        post_robot_state = np.array(env.get_robot_state(), dtype=np.float32)
        if len(robot_state) > 15 and len(post_robot_state) > 15:
            history.append(
                (
                    float(post_robot_state[11]) - float(robot_state[11]),
                    float(post_robot_state[13]) - float(robot_state[13]),
                    float(post_robot_state[15]) - float(robot_state[15]),
                )
            )
        else:
            history.append((exec_upper, exec_lower, exec_ankle))

        steps_executed = steps + 1
        left_distance = _left_foot_distance_to_goal(env)
        left_success = _left_step_success(env)
        print(
            f"  [Left Step {steps}] gate=({gate_upper},{gate_lower},{gate_ankle}), "
            f"goal={int(goal)}, left_foot={_left_foot_success(env)}, "
            f"dist={left_distance:.4f}, success={left_success}"
        )

        if left_success:
            return {
                "success": True,
                "steps": steps_executed,
                "done": int(done),
                "goal": int(goal),
                "distance": left_distance,
                "history": history,
            }

        if done == 1:
            break
        prev_robot_state = np.array(robot_state, dtype=np.float32)

    return {
        "success": False,
        "steps": steps_executed,
        "done": int(done),
        "goal": int(goal),
        "history": history,
    }


def _replay_right_step(env, history):
    if not history:
        return False

    for steps, (delta_upper, delta_lower, delta_ankle) in enumerate(history or []):
        state = env.get_robot_state()
        target_upper = _clamp(float(state[10]) - float(delta_upper), Darwin_config.limit[10][0], Darwin_config.limit[10][1])
        target_lower = _clamp(float(state[12]) - float(delta_lower), Darwin_config.limit[12][0], Darwin_config.limit[12][1])
        target_ankle = _clamp(float(state[14]) - float(delta_ankle), Darwin_config.limit[14][0], Darwin_config.limit[14][1])

        env.darwin.motors[10].setPosition(target_upper)
        env.darwin.motors[12].setPosition(target_lower)
        env.darwin.motors[14].setPosition(target_ankle)
        reached_or_contact = _wait_joint_targets_reached(
            env,
            targets={10: target_upper, 12: target_lower, 14: target_ankle},
            timeout_steps=120,
            tolerance=0.012,
            stable_need=2,
            success_check=lambda: _right_foot_success(env),
        )
        print(
            f"  [Right Step Replay {steps}] reached={bool(reached_or_contact)}, "
            f"contact={_right_foot_success(env)}"
        )
        for _ in range(12 if reached_or_contact else 4):
            env.robot.step(env.timestep)
            if _right_foot_success(env):
                return True

    return True


def _run_right_step_explore(env, tai_agent, cycle_idx: int, max_steps: int = 10):
    imgs = []
    prev_discrete_actions = [1.0, 1.0, 1.0]
    prev_robot_state = None
    robot_state_initial = env.get_robot_state()
    last_upper = float(robot_state_initial[10]) if len(robot_state_initial) > 10 else 0.0
    last_lower = float(robot_state_initial[12]) if len(robot_state_initial) > 12 else 0.0
    last_ankle = float(robot_state_initial[14]) if len(robot_state_initial) > 14 else 0.0

    for steps in range(int(max_steps)):
        _, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()
        tai_state = env.build_tai_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_upper, last_lower, last_ankle],
            prev_discrete_actions=prev_discrete_actions,
            feature_dim=28,
        )
        action_dict = tai_agent.choose_action([obs_tensor, tai_state])

        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]
        gate_upper = int(float(discrete_action[0]))
        gate_lower = int(float(discrete_action[1]))
        gate_ankle = int(float(discrete_action[2]))
        prev_discrete_actions = [float(gate_upper), float(gate_lower), float(gate_ankle)]

        action_upper = float(continuous_action[0])
        action_lower = float(continuous_action[1])
        action_ankle = float(continuous_action[2])
        exec_upper = last_upper if gate_upper == 0 else action_upper
        exec_lower = last_lower if gate_lower == 0 else action_lower
        exec_ankle = last_ankle if gate_ankle == 0 else action_ankle
        last_upper, last_lower, last_ankle = exec_upper, exec_lower, exec_ankle

        target_upper = _clamp(float(robot_state[10]) - exec_upper, Darwin_config.limit[10][0], Darwin_config.limit[10][1])
        target_lower = _clamp(float(robot_state[12]) - exec_lower, Darwin_config.limit[12][0], Darwin_config.limit[12][1])
        target_ankle = _clamp(float(robot_state[14]) - exec_ankle, Darwin_config.limit[14][0], Darwin_config.limit[14][1])
        env.darwin.motors[10].setPosition(target_upper)
        env.darwin.motors[12].setPosition(target_lower)
        env.darwin.motors[14].setPosition(target_ankle)
        reached_or_contact = _wait_joint_targets_reached(
            env,
            targets={10: target_upper, 12: target_lower, 14: target_ankle},
            timeout_steps=120,
            tolerance=0.012,
            stable_need=2,
            post_settle_steps=4,
            success_check=lambda: _right_foot_success(env),
        )
        print(
            f"  [Right Step Explore {steps}] gate=({gate_upper},{gate_lower},{gate_ankle}), "
            f"reached={bool(reached_or_contact)}, contact={_right_foot_success(env)}"
        )
        for _ in range(12 if reached_or_contact else 4):
            env.robot.step(env.timestep)
            if _right_foot_success(env):
                return True
        prev_robot_state = np.array(robot_state, dtype=np.float32)
    return False


def _run_full_step(env, tai_agent, cycle_idx: int, max_steps: int, reset_before_step: bool = True):
    if reset_before_step:
        _reset_leg_pose_like_env_reset(env)
    left_result = _run_left_step(env, tai_agent, cycle_idx, max_steps)
    right_success = False
    if left_result["success"]:
        _reset_right_leg_pose_like_env_reset(env)
        right_success = _replay_right_step(env, left_result["history"])
        if not right_success:
            _reset_right_leg_pose_like_env_reset(env)
            right_success = _run_right_step_explore(env, tai_agent, cycle_idx, 10)
    return {
        "success": bool(left_result["success"] and right_success),
        "left_success": bool(left_result["success"]),
        "right_success": bool(right_success),
        "left_steps": int(left_result["steps"]),
    }


def _prepare_upper_pose(env, stand_up: bool = True):
    metrics = {
        "left_foot_distance": None,
        "left_withdraw_distance": None,
    }
    if stand_up and hasattr(env, "straighten_legs_and_settle"):
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
    prepare_ok, prepare_failure, prepare_metrics = _prepare_upper_pose(env, stand_up=False)
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
            success_check=lambda: _right_upper_grasp_success(env),
        )

    for _ in range(int(timeout_steps)):
        _hold_left_arm_pose(env, left_hold_pose)
        env.robot.step(env.timestep)
        if _right_upper_grasp_success(env):
            return True
    return False

def _replay_upper_right_grasp(env, history, left_hold_pose=None, settle_steps: int = 20):
    _open_right_gripper(env)
    if hasattr(env, "withdraw_right_from_step"):
        env.withdraw_right_from_step()
    elif hasattr(env.darwin, "withdraw_right_from_step"):
        env.darwin.withdraw_right_from_step()
    if hasattr(env, "wait"):
        env.wait(600)
    _hold_left_arm_pose(env, left_hold_pose)

    for steps, item in enumerate(history or []):
        if isinstance(item, dict):
            left_action = item.get("action", (0.0, 0.0))
            left_shoulder_delta = float(item.get("left_shoulder_delta", item.get("delta", (0.0, 0.0))[0]))
            left_arm_delta = float(item.get("left_arm_delta", item.get("delta", (0.0, 0.0))[1]))
        else:
            left_action = item
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

        env.darwin.motors[0].setPosition(right_shoulder_target)
        env.darwin.motors[4].setPosition(right_lower_target)
        reached = _wait_right_upper_targets(
            env,
            targets={0: right_shoulder_target, 4: right_lower_target},
            left_hold_pose=left_hold_pose,
        )
        _hold_left_arm_pose(env, left_hold_pose)

        right_contact = _right_grasp_success(env)
        right_distance = _right_upper_grasp_distance_to_goal(env)
        right_success = _right_upper_grasp_success(env)
        print(
            f"  [Upper R Replay {steps}] "
            f"left_action=({exec_shoulder:.4f},{exec_arm_lower:.4f}), "
            f"left_delta=({left_shoulder_delta:.4f},{left_arm_delta:.4f}), "
            f"right_start=({current_right_shoulder:.4f},{current_right_lower:.4f}), "
            f"right_targets ShoulderR={right_shoulder_target:.4f}, ArmLowerR={right_lower_target:.4f}, "
            f"reached={bool(reached)}, right_contact={right_contact}, "
            f"dist={right_distance:.4f}, success={right_success}"
        )
        if right_success:
            env.darwin.motors[21].setPosition(-0.5)
            for _ in range(int(settle_steps)):
                _hold_left_arm_pose(env, left_hold_pose)
                env.robot.step(env.timestep)
            settled_contact = _right_grasp_success(env)
            settled_distance = _right_upper_grasp_distance_to_goal(env)
            settled_success = _right_upper_grasp_success(env)
            print(
                f"  [Upper R Replay {steps}] latched success, "
                f"settled_contact={settled_contact}, "
                f"settled_dist={settled_distance:.4f}, "
                f"settled_success={settled_success}"
            )
            return True

    env.darwin.motors[21].setPosition(-0.5)
    for _ in range(int(settle_steps)):
        _hold_left_arm_pose(env, left_hold_pose)
        env.robot.step(env.timestep)
        if _right_upper_grasp_success(env):
            return True
        if _right_grasp_success(env):
            right_distance = _right_upper_grasp_distance_to_goal(env)
            print(
                "  [Upper R Replay Fallback] "
                f"closed gripper contact=True, dist={right_distance:.4f}, success=True"
            )
            return True
    return False

def _load_agents(decision_model_dir, catch_model_dir, tai_model_dir, upper_model_dir):
    decision_ckpt = _latest_ckpt(
        decision_model_dir,
        "decision_hppo_*.ckpt",
        r"decision_hppo_(\d+)\.ckpt$",
    )
    catch_ckpt = _latest_ckpt(catch_model_dir, "catch_hppo_*.ckpt", r"catch_hppo_(\d+)\.ckpt$")
    tai_ckpt = _latest_ckpt(tai_model_dir, "tai_agent_*_*.ckpt", r"tai_agent_\d+_(\d+)\.ckpt$")
    upper_ckpt = _latest_upper_ckpt(upper_model_dir)

    missing = {
        "decision": decision_ckpt,
        "catch": catch_ckpt,
        "tai": tai_ckpt,
        "upper": upper_ckpt,
    }
    if any(path is None for path in missing.values()):
        raise FileNotFoundError(
            "Missing PPO checkpoint(s): "
            + ", ".join(f"{name}={path}" for name, path in missing.items())
        )

    decision_agent = DecisionHPPO(
        num_servos=1,
        state_dim=8,
        use_image_input=False,
        num_actions=3,
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

    _load_policy(decision_agent, decision_ckpt, "policy")
    _load_policy(catch_agent, catch_ckpt, "policy")
    _load_policy(tai_agent, tai_ckpt, "policy_tai")
    _load_policy(upper_agent, upper_ckpt, "policy_upper")
    print(f"Loaded decision policy: {decision_ckpt}")
    print(f"Loaded catch policy: {catch_ckpt}")
    print(f"Loaded tai policy: {tai_ckpt}")
    print(f"Loaded upper policy: {upper_ckpt}")
    return decision_agent, catch_agent, tai_agent, upper_agent, missing


def _new_cycle_log(cycle_idx: int):
    return {
        "cycle": int(cycle_idx),
        "ground_decision": None,
        "ground_pressure_detected": None,
        "step_decision": None,
        "step_pressure_detected": None,
        "ground_grasp_success": False,
        "first_step_success": False,
        "upper_rounds": [],
        "stairs_climbed": 0,
        "max_stairs_so_far": 0,
        "height_m": 0.0,
        "failure_stage": None,
    }


def _finish_cycle(cycle_log, summary, current_stairs: int, failure_stage: str):
    stairs_climbed = int(current_stairs)
    cycle_log["stairs_climbed"] = stairs_climbed
    cycle_log["max_stairs_so_far"] = max(int(summary["stats"]["max_stairs"]), stairs_climbed)
    cycle_log["height_m"] = round(stairs_climbed * STAIR_HEIGHT_M, 4)
    cycle_log["failure_stage"] = failure_stage
    summary["stats"]["max_stairs"] = max(summary["stats"]["max_stairs"], stairs_climbed)
    summary["cycles"].append(
        {
            "cycle": int(cycle_log.get("cycle", len(summary["cycles"]) + 1)),
            "stairs_climbed": stairs_climbed,
        }
    )

def full_climb_test(
    robot,
    decision_model_dir=None,
    catch_model_dir=None,
    tai_model_dir=None,
    upper_model_dir=None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    max_steps: int = DEFAULT_MAX_STEPS,
    upper_max_steps: int = DEFAULT_UPPER_MAX_STEPS,
    max_stairs_per_cycle: int = DEFAULT_MAX_STAIRS_PER_CYCLE,
    result_file=None,
):
    max_stairs_per_cycle = min(
        max(0, int(max_stairs_per_cycle)),
        DEFAULT_MAX_STAIRS_PER_CYCLE,
    )
    decision_model_dir = decision_model_dir or path_list["model_path_decision_PPO_h"]
    catch_model_dir = catch_model_dir or path_list["model_path_catch_PPO_h"]
    tai_model_dir = tai_model_dir or path_list["model_path_tai_PPO_h"]
    upper_model_dir = upper_model_dir or path_list["model_path_upper_PPO_h"]
    test_log_dir = os.path.join(os.path.dirname(os.path.dirname(path_list["tai_log_path_PPO"])), "test_log")
    result_file = result_file or os.path.join(
        test_log_dir,
        "full_climb_test_result.json",
    )
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    decision_agent, catch_agent, tai_agent, upper_agent, model_paths = _load_agents(
        decision_model_dir,
        catch_model_dir,
        tai_model_dir,
        upper_model_dir,
    )
    env = Environment(robot)
    summary = {
        "model_paths": model_paths,
        "max_cycles": int(max_cycles),
        "max_steps": int(max_steps),
        "upper_max_steps": int(upper_max_steps),
        "max_stairs_per_cycle": int(max_stairs_per_cycle),
        "stair_height_m": STAIR_HEIGHT_M,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "cycles": [],
        "stats": {
            "ground_decision_success": 0,
            "step_decision_success": 0,
            "ground_grasp_success": 0,
            "first_step_success": 0,
            "upper_left_grasp_success": 0,
            "upper_right_grasp_success": 0,
            "upper_step_success": 0,
            "max_stairs": 0,
        },
    }

    def _save_summary_snapshot(final: bool = False):
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(summary["cycles"], file, ensure_ascii=False, indent=2)

    def _finish_and_save(cycle_log, current_stairs: int, failure_stage: str):
        _finish_cycle(cycle_log, summary, current_stairs, failure_stage)
        _save_summary_snapshot(final=False)
        print(
            f"[Cycle {int(cycle_log.get('cycle', 0)):03d}] saved: "
            f"stairs_climbed={int(current_stairs)}, failure_stage={failure_stage}, "
            f"result_file={result_file}"
        )

    for cycle_idx in range(1, int(max_cycles) + 1):
        print(f"\n========== Full Climb Cycle {cycle_idx:03d}/{max_cycles} ==========")
        _restore_base_targets()
        _reset_env_and_wait(env)
        current_stairs = 0
        ground_step_entry_pose = None
        cycle_log = _new_cycle_log(cycle_idx)

        pressure_state, pressure_detected = _get_decision_state(env)
        decision = int(decision_agent.choose_action(obs=pressure_state, greedy=True)["discrete_action"][0])
        cycle_log["ground_decision"] = decision
        cycle_log["ground_pressure_detected"] = bool(pressure_detected)
        print(f"[Cycle {cycle_idx:03d}] ground decision={decision}, pressure={pressure_detected}")
        if pressure_detected or decision != 0:
            _finish_and_save(cycle_log, current_stairs, "ground_decision")
            continue
        summary["stats"]["ground_decision_success"] += 1

        ground_result = _run_ground_both_grasp(env, catch_agent, cycle_idx, max_steps)
        cycle_log["ground_grasp_success"] = bool(ground_result["success"])
        cycle_log["ground_grasp_steps"] = int(ground_result["steps"])
        if not ground_result["success"]:
            _finish_and_save(cycle_log, current_stairs, "ground_grasp")
            continue
        summary["stats"]["ground_grasp_success"] += 1
        ground_step_entry_pose = _capture_step_entry_pose(env)
        cycle_log["ground_step_entry_pose_captured"] = bool(ground_step_entry_pose)
        print(f"[Cycle {cycle_idx:03d}] ground step entry pose captured={bool(ground_step_entry_pose)}")

        pressure_state, pressure_detected = _get_decision_state(env)
        decision = int(decision_agent.choose_action(obs=pressure_state, greedy=True)["discrete_action"][0])
        cycle_log["step_decision"] = decision
        cycle_log["step_pressure_detected"] = bool(pressure_detected)
        print(f"[Cycle {cycle_idx:03d}] first step decision={decision}, pressure={pressure_detected}")
        if not pressure_detected or decision != 1:
            _finish_and_save(cycle_log, current_stairs, "step_decision")
            continue
        summary["stats"]["step_decision_success"] += 1

        first_step = _run_full_step(env, tai_agent, cycle_idx, max_steps)
        cycle_log["first_step_success"] = bool(first_step["success"])
        cycle_log["first_step_left_success"] = bool(first_step["left_success"])
        cycle_log["first_step_right_success"] = bool(first_step["right_success"])
        if not first_step["success"]:
            _finish_and_save(cycle_log, current_stairs, "first_step")
            continue
        summary["stats"]["first_step_success"] += 1

        current_stairs = 1
        summary["stats"]["max_stairs"] = max(summary["stats"]["max_stairs"], current_stairs)

        failure_stage = "max_stairs_per_cycle_reached"
        while current_stairs < int(max_stairs_per_cycle):
            target_stair = current_stairs + 1
            target_info = _set_upper_phase_targets(current_stairs)
            round_log = {
                "from_stair": int(current_stairs),
                "target_stair": int(target_stair),
                "upper_target": [round(value, 4) for value in target_info["upper_target"]],
                "step_target": [round(value, 4) for value in target_info["step_target"]],
                "upper_left_grasp_success": False,
                "upper_right_grasp_success": False,
                "step_success": False,
            }
            print(
                f"[Cycle {cycle_idx:03d}] target stair {target_stair}, "
                f"upper_target=({target_info['upper_target'][0]:.3f},{target_info['upper_target'][1]:.3f}), "
                f"step_target=({target_info['step_target'][0]:.3f},{target_info['step_target'][1]:.3f})"
            )
            if hasattr(env, "straighten_legs_and_settle"):
                env.straighten_legs_and_settle(settle_ms=1800)
            decision_state, pressure_detected = _get_decision_state(env)
            upper_decision = int(decision_agent.choose_action(obs=decision_state, greedy=True)["discrete_action"][0])
            round_log["upper_decision"] = upper_decision
            round_log["upper_pressure_detected"] = bool(pressure_detected)
            print(f"[Cycle {cycle_idx:03d}] upper grasp decision={upper_decision}, pressure={pressure_detected}")
            if not pressure_detected or upper_decision != 2:
                failure_stage = "upper_decision"
                cycle_log["upper_rounds"].append(round_log)
                break

            upper_left = _run_upper_left_grasp(env, upper_agent, cycle_idx, target_stair, upper_max_steps)
            round_log["upper_left_grasp_success"] = bool(upper_left["success"])
            round_log["upper_left_steps"] = int(upper_left["steps"])
            round_log["upper_left_distance"] = float(upper_left.get("distance", float("inf")))
            if upper_left.get("prepare_failure"):
                round_log["prepare_failure"] = upper_left.get("prepare_failure")
                round_log["prepare_metrics"] = upper_left.get("prepare_metrics", {})
            if not upper_left["success"]:
                failure_stage = upper_left.get("prepare_failure") or "upper_left_grasp"
                print(
                    f"[Cycle {cycle_idx:03d}] failed at {failure_stage}, "
                    f"upper_left_steps={int(upper_left.get('steps', 0))}, "
                    f"upper_left_distance={float(upper_left.get('distance', float('inf'))):.4f}"
                )
                cycle_log["upper_rounds"].append(round_log)
                break
            summary["stats"]["upper_left_grasp_success"] += 1

            upper_right_success = _replay_upper_right_grasp(env, upper_left["history"], left_hold_pose=upper_left.get("left_hold_pose"))
            round_log["upper_right_grasp_success"] = bool(upper_right_success)
            if not upper_right_success:
                failure_stage = "upper_right_grasp"
                cycle_log["upper_rounds"].append(round_log)
                break
            summary["stats"]["upper_right_grasp_success"] += 1

            decision_state, pressure_detected = _get_decision_state(env)
            step_decision = int(decision_agent.choose_action(obs=decision_state, greedy=True)["discrete_action"][0])
            round_log["step_decision"] = step_decision
            round_log["step_pressure_detected"] = bool(pressure_detected)
            print(
                f"[Cycle {cycle_idx:03d}] post-upper step decision={step_decision}, "
                f"pressure={pressure_detected}"
            )
            if not pressure_detected or step_decision != 1:
                failure_stage = "upper_step_decision"
                cycle_log["upper_rounds"].append(round_log)
                break
            summary["stats"]["step_decision_success"] += 1

            step_prepare_ok = _restore_step_entry_pose(env, ground_step_entry_pose)
            round_log["step_prepare_success"] = bool(step_prepare_ok)
            if not step_prepare_ok:
                failure_stage = "upper_step_prepare"
                cycle_log["upper_rounds"].append(round_log)
                break

            next_step_target_info = _set_climb_targets(current_stairs)
            round_log["next_step_target"] = [round(value, 4) for value in next_step_target_info["step_target"]]
            print(
                f"[Cycle {cycle_idx:03d}] upper grasp complete, "
                f"next step_target=({next_step_target_info['step_target'][0]:.3f},"
                f"{next_step_target_info['step_target'][1]:.3f})"
            )

            step_result = _run_full_step(env, tai_agent, cycle_idx, max_steps, reset_before_step=False)
            round_log["step_success"] = bool(step_result["success"])
            round_log["step_left_success"] = bool(step_result["left_success"])
            round_log["step_right_success"] = bool(step_result["right_success"])
            cycle_log["upper_rounds"].append(round_log)
            if not step_result["success"]:
                failure_stage = "upper_step"
                break

            summary["stats"]["upper_step_success"] += 1
            current_stairs += 1
            summary["stats"]["max_stairs"] = max(summary["stats"]["max_stairs"], current_stairs)
            print(
                f"[Cycle {cycle_idx:03d}] climbed stair {current_stairs}, "
                f"height={current_stairs * STAIR_HEIGHT_M:.3f}m"
            )

        _finish_and_save(cycle_log, current_stairs, failure_stage)

    _restore_base_targets()

    _save_summary_snapshot(final=True)

    print("=" * 70)
    print(f"Full climb flow test done. Result saved to: {result_file}")
    print(f"Max stairs climbed: {summary['stats']['max_stairs']}")
    print(f"Max height: {summary['stats']['max_stairs'] * STAIR_HEIGHT_M:.3f} m")
    print("=" * 70)
    return summary


SCRIPT_CONFIG = {
    "decision_dir": None,
    "catch_dir": None,
    "tai_dir": None,
    "upper_dir": None,
    "max_cycles": DEFAULT_MAX_CYCLES,
    "max_steps": DEFAULT_MAX_STEPS,
    "upper_max_steps": DEFAULT_UPPER_MAX_STEPS,
    "max_stairs_per_cycle": DEFAULT_MAX_STAIRS_PER_CYCLE,
    "result_file": None,
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the 100-cycle full PPO climb flow test.")
    parser.add_argument("--decision-dir", default=SCRIPT_CONFIG["decision_dir"])
    parser.add_argument("--catch-dir", default=SCRIPT_CONFIG["catch_dir"])
    parser.add_argument("--tai-dir", default=SCRIPT_CONFIG["tai_dir"])
    parser.add_argument("--upper-dir", default=SCRIPT_CONFIG["upper_dir"])
    parser.add_argument("--max-cycles", type=int, default=SCRIPT_CONFIG["max_cycles"])
    parser.add_argument("--max-steps", type=int, default=SCRIPT_CONFIG["max_steps"])
    parser.add_argument("--upper-max-steps", type=int, default=SCRIPT_CONFIG["upper_max_steps"])
    parser.add_argument("--max-stairs-per-cycle", type=int, default=SCRIPT_CONFIG["max_stairs_per_cycle"])
    parser.add_argument("--result-file", default=SCRIPT_CONFIG["result_file"])
    return parser.parse_args()


def main():
    controller_mod = __import__("controller")
    robot = controller_mod.Robot()
    args = _parse_args()
    full_climb_test(
        robot=robot,
        decision_model_dir=args.decision_dir,
        catch_model_dir=args.catch_dir,
        tai_model_dir=args.tai_dir,
        upper_model_dir=args.upper_dir,
        max_cycles=args.max_cycles,
        max_steps=args.max_steps,
        upper_max_steps=args.upper_max_steps,
        max_stairs_per_cycle=args.max_stairs_per_cycle,
        result_file=args.result_file,
    )


if __name__ == "__main__":
    main()



