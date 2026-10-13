import math
from python_scripts.PPO.utils.Project_config import gps_goal
from python_scripts.PPO.utils.utils import (
    clear_runtime_cache,
    safe_save_checkpoint,
    to_float,
)


def PPO_catch_episoid(
    existing_env,
    episode: int,
    catch_agent,
    log_writer_catch=None,
    log_file_latest_catch=None,
    catch_checkpoint_dir=None,
    train_catch: bool = True,
    save_catch_checkpoint: bool = True,
    min_episodes_to_finish: int = 10000,
    close_gripper_after_step: int = 19,
):
    """执行一次地面抓取训练回合。
    流程1仅负责地面抓取的强化学习循环。
    关节执行、爪子闭合、传感器检测和底层运动控制均在
    Environment.step(...)中处理，该方法委托给RobotRun1执行。
    """
    if existing_env is None:
        raise ValueError("PPO_catch_episoid需要一个现有的 Environment。")

    env = existing_env
    log_enabled = episode <= min_episodes_to_finish
    episode_success = False
    return_all = 0.0
    goal = 0
    steps = 0
    prev_distance = None
    prev_shoulder_action = 0.0
    prev_arm_action = 0.0
    prev_discrete_actions = [1.0, 1.0]
    prev_robot_state = None
    imgs = []

    if log_writer_catch is not None and log_enabled:
        log_writer_catch.add(episode_num=episode)

    print(f"<<<<<<<<<第{episode}周期")
    env.reset()
    env.wait(500)

    _, obs_tensor = env.get_img(steps)
    robot_state = env.get_robot_state()
    robot_state = env.validate_and_clean_data(robot_state)
    catch_state = env.build_catch_feature_state(
        robot_state=robot_state,
        prev_robot_state=prev_robot_state,
        prev_exec_actions=[prev_shoulder_action, prev_arm_action],
        prev_discrete_actions=prev_discrete_actions,
    )
    print("____________________")

    while True:
        obs = (obs_tensor, catch_state)
        action_dict = catch_agent.choose_action(obs=obs)

        discrete_action = action_dict["discrete_action"]
        action_shoulder = action_dict["continuous_action"][0]
        action_arm = action_dict["continuous_action"][1]
        value = action_dict["value"]

        d0 = float(discrete_action[0])
        d1 = float(discrete_action[1])
        cur_shoulder = to_float(action_shoulder)
        cur_arm = to_float(action_arm)

        exec_shoulder = prev_shoulder_action if int(d0) == 0 else cur_shoulder
        exec_arm = prev_arm_action if int(d1) == 0 else cur_arm
        print(
            f"第{episode}周期，第{steps}步, 离散数值：({int(d0)},{int(d1)}), "
            f"连续: ({cur_shoulder:.4f},{cur_arm:.4f}), "
            f"实际: ({exec_shoulder:.4f},{exec_arm:.4f})"
        )

        catch_flag = 1.0 if steps >= int(close_gripper_after_step) else 0.0

        if log_writer_catch is not None and log_enabled:
            log_writer_catch.add_action_catch(action_shoulder, action_arm)

        next_state, done, good, new_goal, _ = env.step(
            robot_state,
            exec_shoulder,
            exec_arm,
            steps,
            catch_flag,
        )
        goal = new_goal
        prev_shoulder_action = exec_shoulder
        prev_arm_action = exec_arm

        print(f"catch_flag: {catch_flag}")
        print(f"done: {done}")

        gps_values = env.validate_and_clean_data(env.print_gps())
        left_gps = gps_values[0] if len(gps_values) > 0 else [0.0, 0.0, 0.0]
        hand_y = float(left_gps[1]) if len(left_gps) > 1 else 0.0
        hand_z = float(left_gps[2]) if len(left_gps) > 2 else 0.0
        if isinstance(gps_goal, (list, tuple)) and len(gps_goal) >= 2:
            goal_y = float(gps_goal[0])
            goal_z = float(gps_goal[1])
        else:
            goal_y = 0.0
            goal_z = 0.0
        dy = goal_y - hand_y
        dz = goal_z - hand_z
        current_distance = math.sqrt(dy ** 2 + dz ** 2)
        print(f"[抓取阶段-动作后] distance_to_goal: {current_distance:.4f}")

        if prev_distance is not None:
            distance_reward = (prev_distance - current_distance) * 15.0
        else:
            distance_reward = -current_distance
        prev_distance = current_distance

        proximity_reward = max(0.0, (0.5 - current_distance) * 5.0)
        action_magnitude = (abs(cur_shoulder) + abs(cur_arm)) / 2.0
        inactivity_penalty = -0.2 if action_magnitude < 0.05 else 0.0
        if int(d0) == 0 and int(d1) == 0:
            inactivity_penalty -= 0.3
        large_action_penalty = -0.05 * (abs(cur_shoulder) > 0.9 or abs(cur_arm) > 0.9)
        reward = distance_reward + proximity_reward + inactivity_penalty + large_action_penalty - 0.5 * steps

        all_grasp_sensors, left_any, right_any = env.read_ground_grasp_state()
        print(f"[抓取阶段] grasp sensors: {all_grasp_sensors}")
        success_flag = int(left_any or right_any)
        if success_flag == 1:
            if current_distance <= 0.04:
                reward += 50.0
                episode_success = True
                print("抓到目标梯级，发放抓取奖励。")
            else:
                reward -= 15.0
                print("抓到非目标梯级，扣除偏差惩罚。")

        if done == 1 and steps <= 2 and success_flag != 1:
            print("因环境不稳定导致无效数据，跳过此抓取回合。")
            break

        return_all += reward
        steps += 1
        next_obs_img, next_obs_tensor = env.get_img(steps)

        if good == 1:
            next_robot_state = env.validate_and_clean_data(next_state)
            next_catch_state = env.build_catch_feature_state(
                robot_state=next_robot_state,
                prev_robot_state=robot_state,
                prev_exec_actions=[exec_shoulder, exec_arm],
                prev_discrete_actions=[d0, d1],
            )
            exec_log_probs = catch_agent.evaluate_action_log_probs(
                obs,
                discrete_action,
                [cur_shoulder, cur_arm],
            )
            catch_agent.store_transition(
                state=[obs_tensor, catch_state],
                discrete_action=discrete_action,
                continuous_action=[cur_shoulder, cur_arm],
                reward=reward,
                next_state=[
                    next_obs_img,
                    next_catch_state,
                ],
                done=done,
                value=value,
                discrete_log_prob=exec_log_probs["discrete_log_prob"],
                continuous_log_prob=exec_log_probs["continuous_log_prob"],
            )

        prev_robot_state = robot_state
        robot_state = env.get_robot_state()
        robot_state = env.validate_and_clean_data(robot_state)
        catch_state = env.build_catch_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[exec_shoulder, exec_arm],
            prev_discrete_actions=[d0, d1],
        )
        obs_tensor = next_obs_tensor

        if (
            save_catch_checkpoint
            and catch_checkpoint_dir
            and episode > 100
            and episode <= min_episodes_to_finish
            and done == 1
            and episode % 500 == 0
        ):
            save_path = f"{catch_checkpoint_dir}/catch_hppo_{episode}.ckpt"
            checkpoint = {
                "policy": catch_agent.policy.state_dict(),
                "optimizer_hppo": catch_agent.optimizer.state_dict(),
                "episode": episode,
            }
            safe_save_checkpoint(save_path, checkpoint)

        episode_done = bool(done == 1 or catch_flag == 1.0)

        if episode_done and steps > 3:
            if train_catch and episode <= min_episodes_to_finish:
                loss_actor, loss_critic = catch_agent.learn()
                if loss_actor != 0 and loss_critic != 0:
                    print("loss_actor:", loss_actor, "loss_critic:", loss_critic)
                    if log_writer_catch is not None and log_enabled:
                        log_writer_catch.add_loss_hppo_catch_actor_critic(loss_actor, loss_critic)
                        if log_file_latest_catch:
                            try:
                                log_writer_catch.save_catch(log_file_latest_catch)
                            except Exception as e:
                                print(f"保存抓取loss到日志失败: {e}")
            elif train_catch:
                print(f"抓取模型训练已超过 {min_episodes_to_finish} 周期，跳过 learn()。")
            else:
                print("抓取模型训练已禁用，跳过 learn()。")

            if log_writer_catch is not None and log_enabled:
                log_writer_catch.add(return_all=return_all)
                log_writer_catch.add(goal=goal)

        if episode_done:
            env.wait(100)
            clear_runtime_cache(imgs)
            episode += 1

            if episode_success:
                print("【抓取成功】保持机器人抓取状态，准备回到流程0进行下一次决策。")
                env.wait(200)

            break

    if log_writer_catch is not None and log_file_latest_catch and log_enabled:
        log_writer_catch.save_catch(log_file_latest_catch)

    if not episode_success:
        print("Ground grasp did not catch the target step; reset environment.")
        env.reset()
        env.wait(500)
        clear_runtime_cache(imgs)

    return episode_success, episode
