import math
import os
import numpy as np
from python_scripts.PPO.utils.Project_config import gps_goal2

from python_scripts.PPO.utils.utils import (
    clear_runtime_cache,
    safe_save_checkpoint,
)


def PPO_upper_episoid(
    existing_env=None,
    total_episode=0,
    episode=0,
    log_writer_upper=None,
    log_file_latest_upper=None,
    upper_agent=None,
    save_upper_checkpoint: bool = True,
    upper_checkpoint_dir=None,
    min_episodes_to_finish: int = 10000,
):

    env = existing_env
    log_enabled = episode <= min_episodes_to_finish

    if log_writer_upper is not None and log_enabled:
        log_writer_upper.add(episode_num=episode)


    steps = 0
    done = 0
    goal = 0
    return_all = 0.0
    prev_robot_state = None
    prev_target_dist = None

    robot_state = env.get_robot_state()
    last_action_shoulder = robot_state[1] if len(robot_state) > 1 else 0.0
    last_action_arm_lower = robot_state[5] if len(robot_state) > 5 else 0.0
    prev_discrete_actions = [1.0, 1.0]

    imgs = []

    while True:
        _, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()

        upper_state = env.build_upper_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_action_shoulder, last_action_arm_lower],
            prev_discrete_actions=prev_discrete_actions,
        )
        obs = (obs_tensor, upper_state)
        upper_dict = upper_agent.choose_action(obs=obs)
        gate_s = int(upper_dict["discrete_action"][0])
        gate_u = int(upper_dict["discrete_action"][1])

        prev_discrete_actions = [float(gate_s), float(gate_u)]

        act_s = float(upper_dict["continuous_action"][0])
        act_l = float(upper_dict["continuous_action"][1])

        exec_s = last_action_shoulder if gate_s == 0 else act_s
        exec_l = last_action_arm_lower if gate_u == 0 else act_l

        last_action_shoulder = exec_s
        last_action_arm_lower = exec_l

        _, _, done_env, _, goal_env, _ = env.step3(
            robot_state,
            exec_s,
            exec_l,
            steps,
        )

        env.wait(420)

        hand_pressures = [
            env.safe_touch("grasp_L1"),
            env.safe_touch("grasp_L1_1"),
            env.safe_touch("grasp_L1_2"),
        ]
        left_any = any(v > 0.5 for v in hand_pressures[:3])

        left_gps = env.print_gps()[0]
        left_y = float(left_gps[1]) if len(left_gps) > 1 else 0.0
        left_z = float(left_gps[2]) if len(left_gps) > 2 else 0.0

        dy = gps_goal2[0] - left_y
        dz = gps_goal2[1] - left_z
        target_dist = math.sqrt(dy * dy + dz * dz)

        if prev_target_dist is not None:
            distance_reward = (prev_target_dist - target_dist) * 15.0
        else:
            distance_reward = -target_dist
        prev_target_dist = target_dist

        proximity_reward = max(0.0, (0.5 - target_dist) * 5.0)
        action_magnitude = (abs(act_s) + abs(act_l)) / 2.0
        inactivity_penalty = -0.2 if action_magnitude < 0.05 else 0.0
        if int(gate_s) == 0 and int(gate_u) == 0:
            inactivity_penalty -= 0.3
        large_action_penalty = -0.05 * (abs(act_s) > 0.9 or abs(act_l) > 0.9)
        reward = distance_reward + proximity_reward + inactivity_penalty + large_action_penalty - 0.5 * steps

        left_close_on_target = left_any and (target_dist <= 0.15)
        success_grasp = False
        if left_any:
            if target_dist <= 0.04:
                reward += 50.0
                success_grasp = True
            else:
                reward -= 15.0

        done = 0
        if success_grasp:
            goal = 1
            done = 1
        else:
            goal = 0

        if steps >= 19:
            done = 1

        print(
            f"第{episode}周期，第{steps}步, 离散数值：({int(gate_s)},{int(gate_u)}), "
            f"连续: ({act_s:.4f},{act_l:.4f}), "
            f"实际: ({exec_s:.4f},{exec_l:.4f})"
        )
        print(
            f"[Upper状态] step={steps + 1}/20, done_env={int(done_env)}, done={int(done)}, goal={int(goal)}, "
            f"left_close={int(left_close_on_target)}, success_grasp={int(success_grasp)}, "
            f"target_dist={target_dist:.4f}, reward={reward:.4f}"
        )

        return_all += reward

        if np.isfinite(reward):
            next_robot_state = env.get_robot_state()
            next_upper_state = env.build_upper_feature_state(
                robot_state=next_robot_state,
                prev_robot_state=robot_state,
                prev_exec_actions=[exec_s, exec_l],
                prev_discrete_actions=prev_discrete_actions,
            )

            exec_log_probs = upper_agent.evaluate_action_log_probs(
                obs,
                [gate_s, gate_u],
                [act_s, act_l],
            )
            upper_agent.store_transition(
                state=[obs_tensor, upper_state],
                discrete_action=[gate_s, gate_u],
                continuous_action=[act_s, act_l],
                reward=reward,
                next_state=[obs_tensor, next_upper_state],
                done=done,
                value=upper_dict["value"],
                discrete_log_prob=exec_log_probs["discrete_log_prob"],
                continuous_log_prob=exec_log_probs["continuous_log_prob"],
            )

        prev_robot_state = np.array(robot_state, dtype=np.float32)
        steps += 1

        if done == 1:
            if episode <= min_episodes_to_finish:
                loss_actor, loss_critic = upper_agent.learn()
                if loss_actor != 0 and loss_critic != 0:
                    total_loss = loss_actor + loss_critic
                    if log_writer_upper is not None and log_enabled:
                        log_writer_upper.add(return_all=return_all)
                        log_writer_upper.add(goal=goal)
                        log_writer_upper.add(loss=total_loss)
                        log_writer_upper.add(loss_actor=loss_actor)
                        log_writer_upper.add(loss_critic=loss_critic)
                        if log_file_latest_upper:
                            log_writer_upper.save_tai(log_file_latest_upper)
                            try:
                                log_writer_upper.clear()
                            except Exception:
                                pass
                elif log_writer_upper is not None and log_enabled:
                    log_writer_upper.add(return_all=return_all)
                    log_writer_upper.add(goal=goal)
            else:
                print(f"台阶上抓取训练已超过 {min_episodes_to_finish} 周期，跳过 learn()。")

            if save_upper_checkpoint and episode <= min_episodes_to_finish and episode % 500 == 0 and episode > 100:
                save_path = os.path.join(upper_checkpoint_dir, f"upper_agent_{total_episode}_{episode}.ckpt")
                ckpt = {
                    "episode": episode,
                    "policy_upper": upper_agent.policy.state_dict(),
                    "optimizer_upper": upper_agent.optimizer.state_dict(),
                }
                safe_save_checkpoint(save_path, ckpt)

            env.reset()
            env.wait(600)
            clear_runtime_cache(imgs)
            break

    return bool(goal)
