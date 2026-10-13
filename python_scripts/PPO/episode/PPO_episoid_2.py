import math
import numpy as np
from python_scripts.PPO.utils.Project_config import gps_goal1
from python_scripts.PPO.utils.utils import (
    clear_runtime_cache,
    safe_save_checkpoint,
)


def PPO_tai_episoid(
    existing_env=None,
    total_episode=0,
    episode=0,
    log_writer_tai=None,
    log_file_latest_tai=None,
    tai_agent=None,
    keep_pose_on_success=False,
    train_tai=True,
    save_tai_checkpoint=True,
    tai_checkpoint_dir=None,
    min_episodes_to_finish: int = 10000
):

    log_enabled = episode <= min_episodes_to_finish
    env = existing_env
    print("开始抬腿！")
    env.darwin.tai_leg_L1()
    env.darwin.tai_leg_L2()

    # 初始化状态
    return_all = 0
    prev_potential = None
    prev_center_error = None
    imgs = []
    goal = 0
    done = 0
    steps = 0
    episode_success = False
    left_step_history = []

    # 初始化用于保持姿势的变量
    last_action_LegUpper = 0.0
    last_action_LegLower = 0.0
    last_action_Ankle = 0.0
    prev_discrete_actions = [1.0, 1.0, 1.0]
    prev_robot_state = None

    # 获取观察和状态
    print("____________________")

    # 记录回合数
    if log_writer_tai is not None and log_enabled:
        log_writer_tai.add(episode_num=episode)

    while True:
        _, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()
        robot_state = env.validate_and_clean_data(robot_state)
        tai_state = env.build_tai_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=[last_action_LegUpper, last_action_LegLower, last_action_Ankle],
            prev_discrete_actions=prev_discrete_actions,
            feature_dim=28,
        )
        # 选择动作 - 使用tai_agent替代三个独立的PPO2智能体
        obs = (obs_tensor, tai_state)
        tai_dict = tai_agent.choose_action(obs=obs)

        tai_discrete_action = tai_dict['discrete_action']  # [3] - 三个舵机的离散动作
        tai_continuous_action = tai_dict['continuous_action']  # [3] - 三个舵机的连续动作
        tai_continuous_log_prob = tai_dict['continuous_log_prob']  # [3] - 连续动作的对数概率
        tai_value = tai_dict['value']  # 状态价值

        # 提取三个关节的连续动作和对应的离散动作
        action_LegUpper = float(tai_continuous_action[0])  # 上腿关节
        action_LegLower = float(tai_continuous_action[1])  # 下腿关节
        action_Ankle = float(tai_continuous_action[2])  # 踝关节

        # 提取离散动作（门控信号）
        discrete_upper = int(tai_discrete_action[0])
        discrete_lower = int(tai_discrete_action[1])
        discrete_ankle = int(tai_discrete_action[2])
        prev_discrete_actions = [float(discrete_upper), float(discrete_lower), float(discrete_ankle)]

        # 提取对应的对数概率
        log_prob_LegUpper = float(tai_continuous_log_prob[0])
        log_prob_LegLower = float(tai_continuous_log_prob[1])
        log_prob_Ankle = float(tai_continuous_log_prob[2])

        action_LegUpper_original = action_LegUpper
        action_LegLower_original = action_LegLower
        action_Ankle_original = action_Ankle

        # 根据离散动作决定是使用新动作还是保持上一动作
        if discrete_upper == 0:
            action_LegUpper_exec = last_action_LegUpper
        else:
            action_LegUpper_exec = action_LegUpper

        if discrete_lower == 0:
            action_LegLower_exec = last_action_LegLower
        else:
            action_LegLower_exec = action_LegLower

        if discrete_ankle == 0:
            action_Ankle_exec = last_action_Ankle
        else:
            action_Ankle_exec = action_Ankle

        # 更新 "last_action" 变量以备下一步使用
        last_action_LegUpper = action_LegUpper_exec
        last_action_LegLower = action_LegLower_exec
        last_action_Ankle = action_Ankle_exec

        # 分别添加动作、对数概率和状态价值到日志
        value_scalar = tai_value if isinstance(tai_value, (int, float)) else tai_value.item()
        if log_writer_tai is not None and log_enabled:
            log_writer_tai.add_action_tai(action_LegUpper, action_LegLower, action_Ankle)
            log_writer_tai.add_log_prob_tai(log_prob_LegUpper, log_prob_LegLower, log_prob_Ankle)
            log_writer_tai.add_value_tai(value_scalar, value_scalar, value_scalar)

        print("第", steps + 1, "步")
        print(f"【tai_agent门控控制】离散动作: [{discrete_upper}, {discrete_lower}, {discrete_ankle}]")
        print(
            f"【原始连续动作】LegUpper: {action_LegUpper_original:.4f}, LegLower: {action_LegLower_original:.4f}, Ankle: {action_Ankle_original:.4f}")
        print(
            f"【最终执行动作】LegUpper: {action_LegUpper_exec:.4f}, LegLower: {action_LegLower_exec:.4f}, Ankle: {action_Ankle_exec:.4f}")

        # 获取GPS数据
        gps_values = env.print_gps()
        gps_values = env.validate_and_clean_data(gps_values)
        foot_gps = gps_values[4] if len(gps_values) > 4 else [0.0, 0.0, 0.0]
        foot_y_dbg = float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
        foot_z_dbg = float(foot_gps[2]) if len(foot_gps) > 2 else 0.0
        print(f"[动作前] 脚部位置传感器(foot_gps1): {foot_gps}, 脚部落点(y,z)=({foot_y_dbg:.4f}, {foot_z_dbg:.4f})")

        # 执行动作
        _, _, done, good, goal, _ = env.step2(robot_state, action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec, steps)

        # 关键：这里是动作执行完成后的实时脚部位置，和最终踩踏状态一致。
        gps_values_post = env.validate_and_clean_data(env.print_gps())
        foot_gps_post = gps_values_post[4] if len(gps_values_post) > 4 else [0.0, 0.0, 0.0]
        foot_y_post = float(foot_gps_post[1]) if len(foot_gps_post) > 1 else 0.0
        foot_z_post = float(foot_gps_post[2]) if len(foot_gps_post) > 2 else 0.0
        print(
            f"[动作后] 脚部位置传感器(foot_gps1): {foot_gps_post}, "
            f"脚部落点(y,z)=({foot_y_post:.4f}, {foot_z_post:.4f})"
        )

        # 记录左脚“实际执行增量”（动作后关节 - 动作前关节），
        # 右脚复现时按镜像增量回放，可避免限幅/到位误差导致的左右不一致。
        post_robot_state_for_replay = env.validate_and_clean_data(env.get_robot_state())
        if len(robot_state) > 15 and len(post_robot_state_for_replay) > 15:
            real_du = float(post_robot_state_for_replay[11]) - float(robot_state[11])
            real_dl = float(post_robot_state_for_replay[13]) - float(robot_state[13])
            real_da = float(post_robot_state_for_replay[15]) - float(robot_state[15])
            left_step_history.append((real_du, real_dl, real_da))
        else:
            # 回退：状态异常时保留原有行为，避免中断训练流程。
            left_step_history.append((action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec))

        # ========== 奖励计算（统一使用自定义reward） ==========
        # =========================
        # 新版抬腿/踩台阶 reward（直接写在循环里）
        # 依赖：prev_potential、prev_foot_height 需要在 episode 开始前初始化为 None
        # =========================

        # ========== 奖励计算（改进版：稠密进度奖励 + 成功终止 + 抓持约束） ==========
        reward = 0.0

        foot_y = foot_y_post  # 脚高度（y），使用动作后位置
        foot_z = foot_z_post  # 脚前后位置（z），使用动作后位置

        # 优先使用配置目标，避免硬编码目标与场景不一致
        if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
            goal_y = float(gps_goal1[0])
            goal_z = float(gps_goal1[1])
        else:
            goal_y = 0.058
            goal_z = 0.023

        dy = goal_y - foot_y
        dz = goal_z - foot_z
        dist = math.sqrt(dy ** 2 + dz ** 2)

        # 1) 基础距离惩罚（控制量级，避免主导梯度）
        reward -= dist * 2.0

        # 2) 进度奖励：每步只奖励"比上一步更接近"
        if prev_potential is not None:
            progress = prev_potential - dist
            reward += 80.0 * progress
        prev_potential = dist

        # 3) 中心误差奖励（重点压横向偏差dz，避免踩偏）
        lateral_w = 2.8  # dz权重更大
        center_error = math.sqrt(dy ** 2 + (lateral_w * dz) ** 2)
        reward -= 4.0 * center_error
        reward -= 4.0 * abs(dz)
        if prev_center_error is not None:
            center_progress = prev_center_error - center_error
            reward += 120.0 * center_progress
        prev_center_error = center_error

        # 4) 成功奖励（严格成功：必须在目标阈值内）
        strict_success = (goal == 1 and dist <= 0.03)
        if strict_success:
            reward += 400.0
            print(f"🎉 成功踩踏！dist={dist:.4f} <= 0.03")
            done = 1
            episode_success = True
        elif goal == 1 and dist > 0.03:
            # 兜底：若环境层返回goal=1但距离不达标，按偏踩处理
            reward -= 8.0
            goal = 0
            done = 1
            print(f"⚠️ 偏踩，不计成功。dist={dist:.4f} > 0.03")

        # 5) 失败保护：若脚偏离过远给额外惩罚，加速回合结束
        if dist > 0.25:
            reward -= 3.0

        # 5.1) 偏踩惩罚：接触到台阶但落点不在目标邻域
        # 这里与环境层严格成功条件保持一致，避免“踩到但偏位”被误强化。
        if done == 1 and goal == 0 and dist > 0.03:
            reward -= 8.0

        # 7) 动作惩罚（防抖+平滑）
        act_mag = abs(action_LegUpper_exec) + abs(action_LegLower_exec) + abs(action_Ankle_exec)
        reward -= 0.008 * act_mag

        # 8) 步数惩罚：鼓励更快达成目标
        reward -= 0.012 * steps

        return_all += reward

        steps += 1
        # 获取新的观察
        _, next_obs_tensor = env.get_img(steps)

        # 存储经验（除去明显无效样本）
        should_store = True
        if done == 1 and steps <= 2 and good != 1:
            should_store = False
            print(f"  跳过无效样本：done={done}, steps={steps}, good={good}")
        if not np.isfinite(reward):
            should_store = False
            print(f"  跳过异常样本：reward={reward}")

        if should_store and train_tai:
            next_robot_state = post_robot_state_for_replay
            next_tai_state = env.build_tai_feature_state(
                robot_state=next_robot_state,
                prev_robot_state=robot_state,
                prev_exec_actions=[action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec],
                prev_discrete_actions=prev_discrete_actions,
                feature_dim=28,
            )
            # 使用tai_agent存储转移经验
            exec_log_probs = tai_agent.evaluate_action_log_probs(
                obs,
                tai_discrete_action,
                [action_LegUpper, action_LegLower, action_Ankle],
            )
            tai_agent.store_transition(
                state=[obs_tensor, tai_state],
                discrete_action=tai_discrete_action,
                continuous_action=[action_LegUpper, action_LegLower, action_Ankle],
                reward=reward,
                next_state=[next_obs_tensor, next_tai_state],
                done=done,
                value=tai_value,
                discrete_log_prob=exec_log_probs["discrete_log_prob"],
                continuous_log_prob=exec_log_probs["continuous_log_prob"]
            )
            print(f"  已存储经验: reward={reward:.4f}")
        elif should_store and not train_tai:
            print("  踩踏为推理模式：跳过经验存储。")

        # 更新状态
        prev_robot_state = np.array(robot_state, dtype=np.float32)
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor

        # 检查是否结束当前回合
        if steps >= 19:
            done = 1

        # 定期保存模型
        if save_tai_checkpoint and tai_checkpoint_dir and episode <= min_episodes_to_finish and episode % 500 == 0 and done == 1 and episode > 100:
            try:
                save_path = f"{tai_checkpoint_dir}/tai_agent_{total_episode}_{episode}.ckpt"
                print(f"保存模型到: {save_path}")
                checkpoint = {
                    "episode": episode,
                    "policy_tai": tai_agent.policy.state_dict(),
                    "optimizer_tai": tai_agent.optimizer.state_dict()
                }
                safe_save_checkpoint(save_path, checkpoint)
            except Exception as e:
                print(f"模型保存失败: {e}")

        # 学习过程
        if train_tai and episode > 0 and done == 1:
            # ========== Loss 计算 ==========
            # 使用tai_agent进行学习，返回离散和连续两个损失
            if episode <= min_episodes_to_finish:
                loss_actor, loss_critic = tai_agent.learn()
                if loss_actor != 0 and loss_critic != 0:
                    print("=" * 60)
                    print(f"【第 {episode} 回合训练完成】")
                    print(f"  累积奖励 (return_all): {return_all:.4f}")
                    print(f"   (goal): {goal}")
                    print(f"  actor损失 (loss_actor): {loss_actor:.6f}")
                    print(f"  critic损失 (loss_critic): {loss_critic:.6f}")
                    print(f"  总损失 (total loss): {loss_actor + loss_critic:.6f}")

                    # 记录损失值
                    total_loss = loss_actor + loss_critic
                    if log_writer_tai is not None and log_enabled:
                        log_writer_tai.add(loss=total_loss)
                        log_writer_tai.add(loss_actor=loss_actor)
                        log_writer_tai.add(loss_critic=loss_critic)
            else:
                print(f"踩踏模型训练已超过 {min_episodes_to_finish} 周期，跳过 learn()。")

            if log_writer_tai is not None and log_enabled:
                log_writer_tai.add(return_all=return_all)
                log_writer_tai.add(goal=goal)
        elif (not train_tai) and done == 1:
            print("踩踏为推理模式：跳过learn与checkpoint保存。")
            if log_writer_tai is not None and log_enabled:
                log_writer_tai.add(return_all=return_all)
                log_writer_tai.add(goal=goal)

        print("done:", done)

        if done == 1:
            print("抬腿回合结束，重置环境...")
            if keep_pose_on_success and episode_success:
                print("抬腿成功，保留当前姿态进入第三阶段。")
                # 在进入第三阶段前，额外校验左脚当前位置是否仍满足成功距离阈值
                try:
                    gps_now = env.validate_and_clean_data(env.print_gps())
                    foot_now = gps_now[4] if len(gps_now) > 4 else [0.0, 0.0, 0.0]
                    fy = float(foot_now[1]) if len(foot_now) > 1 else 0.0
                    fz = float(foot_now[2]) if len(foot_now) > 2 else 0.0
                    if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
                        gy = float(gps_goal1[0])
                        gz = float(gps_goal1[1])
                    else:
                        gy, gz = 0.058, 0.023
                    cur_dist = math.sqrt((gy - fy) ** 2 + (gz - fz) ** 2)
                    if cur_dist > 0.03:
                        print(f"⚠️ 左脚当前距离目标 {cur_dist:.4f} 超过阈值 0.03，跳过三阶段训练并重置。")
                        env.darwin.robot_reset()
                        env.reset()
                        clear_runtime_cache(imgs)
                        for _ in range(40):
                            try:
                                env.robot.step(env.timestep)
                            except Exception:
                                pass
                        env.wait(1000)
                        return False
                except Exception:
                    print("⚠️ 无法读取左脚GPS以校验位置，跳过三阶段训练并重置。")
                    env.darwin.robot_reset()
                    env.reset()
                    clear_runtime_cache(imgs)
                    for _ in range(40):
                        env.robot.step(env.timestep)
                    env.wait(1000)
                    return False
                env._last_tai_left_step_history = list(left_step_history)
                if log_enabled:
                    log_writer_tai.save_tai(log_file_latest_tai)
                log_writer_tai.clear()  # ✅ 清空
                break

            env.darwin.robot_reset()  # 重置环境
            env.reset()
            del obs_tensor
            del next_obs_tensor
            del prev_potential
            del prev_center_error
            # 增加初始稳定时间
            print("等待稳定...")
            for _ in range(40):  # 增加40个时间步的稳定时间
                env.robot.step(env.timestep)

            print("等待一秒...")
            env.wait(1000)
            imgs = []
            steps = 0

            if log_enabled:
                log_writer_tai.save_tai(log_file_latest_tai)
            log_writer_tai.clear()  #  清空

            break

    return episode_success
