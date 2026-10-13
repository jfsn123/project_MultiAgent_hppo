"""
PPO 第 0 阶段总调度流程。
包含高层决策、地面抓取、踩踏和台阶上抓取四个阶段的串联训练逻辑。
"""
import os

import numpy as np

from python_scripts.PPO.episode.PPO_episoid_1 import PPO_catch_episoid
from python_scripts.PPO.episode.PPO_episoid_2 import PPO_tai_episoid
from python_scripts.PPO.episode.PPO_episoid_3 import PPO_upper_episoid
from python_scripts.PPO.utils.PPO_Log_write import Log_write
from python_scripts.PPO.utils.Webots_interfaces import Environment
from python_scripts.PPO.utils.utils import (
    safe_save_checkpoint,
    clear_runtime_cache,
    initialize_directories,
    initialize_log_files,
    initialize_models,
    make_agents,
    train_and_log_decision,
)


UPPER_GRASP_STEP_DISTANCE_LIMIT = 0.04


def PPO_episoid_0(
    robot,
    decision_model_path=None,
    catch_model_path=None,
    tai_model_path=None,
    upper_model_path=None,
    train_decision: bool = True,
    train_catch: bool = True,
    train_tai: bool = True,
    train_upper: bool = True,
):
    """
    执行 PPO 第 0 阶段的完整训练/推理流程。

    参数:
        robot: Webots 机器人实例。
        decision_model_path: 决策模型路径。
        catch_model_path: 地面抓取模型路径。
        tai_model_path: 踩踏模型路径。
        upper_model_path: 台阶上抓取模型路径。
        train_decision: 是否训练高层决策网络。
        train_catch: 是否训练地面抓取网络。
        train_tai: 是否训练踩踏网络。
        train_upper: 是否训练台阶上抓取阶段。

    返回:
        (False, env)
    """
    # 保存 checkpoint 的开关与各阶段训练开关保持一致。
    save_decision_checkpoint = train_decision
    save_catch_checkpoint = train_catch
    save_tai_checkpoint = train_tai
    save_upper_checkpoint = train_upper

    # ==================== 创建各阶段智能体 ====================
    decision_agent, catch_agent, tai_agent, upper_agent = make_agents()

    # ==================== 创建日志写入器 ====================
    log_writer_decision = Log_write()
    log_writer_catch = Log_write()
    log_writer_tai = Log_write()
    log_writer_upper = Log_write()

    # ==================== 初始化 checkpoint 目录 ====================
    decision_checkpoint_dir, catch_checkpoint_dir, tai_checkpoint_dir, upper_checkpoint_dir = (
        initialize_directories()
    )

    # ==================== 初始化日志文件 ====================
    log_file_latest_decision, log_file_latest_catch, log_file_latest_tai, log_file_latest_upper = (
        initialize_log_files()
    )

    # ==================== 初始化 episode 计数与模型状态 ====================
    decision_episode, catch_episode, tai_episode, upper_episode = initialize_models(
        decision_agent=decision_agent,
        catch_agent=catch_agent,
        tai_agent=tai_agent,
        upper_agent=upper_agent,
        decision_dir=decision_checkpoint_dir,
        catch_dir=catch_checkpoint_dir,
        tai_dir=tai_checkpoint_dir,
        upper_dir=upper_checkpoint_dir,
        decision_model_path=decision_model_path,
        catch_model_path=catch_model_path,
        tai_model_path=tai_model_path,
        upper_model_path=upper_model_path,
    )

    # ==================== 打印训练配置 ====================
    print("\n[Training config]")
    print(f"  - 决策训练: {train_decision}")
    print(f"  - 地面上抓取训练: {train_catch}")
    print(f"  - 踩踏训练: {train_tai}")
    print(f"  - 台阶上抓取训练: {train_upper}\n")
    if not save_decision_checkpoint:
        print("决策模型不会保存")
    if not save_catch_checkpoint:
        print("地面抓取模型不会保存")
    if not save_tai_checkpoint:
        print("踩踏模型不会保存")
    if not save_upper_checkpoint:
        print("台阶上抓取模型不会保存")

    # ==================== 初始化运行时状态 ====================
    correct_decisions = 0
    total_decisions = 0
    catch_success = False
    episodes_to_finish = 10000

    env = Environment(robot)
    log_enabled = decision_episode <= episodes_to_finish

    catch_only_mode = train_catch and not train_decision and not train_tai and not train_upper

    if catch_only_mode:
        # 仅训练地面抓取：当且仅当只开启 train_catch 时进入该分支。
        print("\n仅训练地面上抓取\n")
        while catch_episode <= episodes_to_finish:
            print(f"Catch Episode {catch_episode + 1}")
            _, catch_episode = PPO_catch_episoid(
                existing_env=env,
                episode=catch_episode,
                catch_agent=catch_agent,
                log_writer_catch=log_writer_catch,
                log_file_latest_catch=log_file_latest_catch,
                catch_checkpoint_dir=catch_checkpoint_dir,
                train_catch=train_catch,
                save_catch_checkpoint=save_catch_checkpoint,
                min_episodes_to_finish=episodes_to_finish,
            )
        print(f"地面上抓取训练结束d: catch_episode={catch_episode}")
        return

    # ==================== 主训练循环 ====================
    while True:
        # 当所有阶段都超过 episode 上限时结束。
        if (decision_episode > episodes_to_finish and catch_episode > episodes_to_finish
            and tai_episode > episodes_to_finish and upper_episode > episodes_to_finish):
            print(
                f"决策的周期数={decision_episode}, 抓取的周期数={catch_episode}, "
                f"踩踏的周期数={tai_episode}, 台阶上抓取的周期数={upper_episode} "
                f"(上限: {episodes_to_finish})"
            )
            break

        decision_episode += 1
        print(f"Total Episode {decision_episode}")

        # 每 500 个 episode 保存一次决策网络 checkpoint。
        if (save_decision_checkpoint and decision_episode <= episodes_to_finish and decision_episode % 500 == 0):
            dec_path = os.path.join(decision_checkpoint_dir, f"decision_hppo_{decision_episode}.ckpt")
            dec_ckpt = {
                "policy": decision_agent.policy.state_dict(),
                "optimizer": decision_agent.optimizer.state_dict(),
                "episode": decision_episode,
            }
            safe_save_checkpoint(dec_path, dec_ckpt)

        # ==================== 决策阶段 ====================
        pressure_state, pressure_values, pressure_detected = env.get_hand_pressure_state()
        hand_foot_distance = env.hand_foot_gps_distance()
        foot_y_post = env.get_foot_y_post()
        decision_state = np.concatenate(
            (
                pressure_state,
                np.array([hand_foot_distance, foot_y_post], dtype=np.float32),
            )
        )
        if not pressure_detected:
            expected_decision = 0
        elif hand_foot_distance < 0.3 and foot_y_post > 0.03:
            expected_decision = 2
        else:
            expected_decision = 1

        print(
            f"hand_pressure values: {pressure_values.tolist()}, detected={pressure_detected}, "
            f"left_gps1-foot_gps1 distance={hand_foot_distance:.4f}, "
            f"foot_y_post={foot_y_post:.4f}"
        )

        decision_dict = decision_agent.choose_action(obs=decision_state)
        decision = int(decision_dict["discrete_action"][0])
        decision_reward = 10.0 if decision == expected_decision else -10.0
        print(f"High-level decision = {decision} (0=ground grasp, 1=stepping, 2=upper grasp)")

        if decision == 0:
            # ==================== 决策分支：执行地面抓取 ====================
            if expected_decision != 0:
                print("Decision error: pressure already detected but selected grasp.")
            else:
                print("Decision correct: no hand pressure, selected grasp.")

            if train_decision and log_enabled:
                correct_decisions, total_decisions = train_and_log_decision(
                    decision_agent=decision_agent,
                    log_writer_decision=log_writer_decision,
                    log_file_latest_decision=log_file_latest_decision,
                    total_episode=decision_episode,
                    decision=decision,
                    decision_reward=decision_reward,
                    decision_dict=decision_dict,
                    d_decision_state=decision_state,
                    correct_decisions=correct_decisions,
                    total_decisions=total_decisions,
                )
            elif train_decision:
                print(f"Decision training exceeded {episodes_to_finish} episodes; inference only.")
            else:
                print("Decision training disabled; inference only.")

            if expected_decision != 0:
                # 如果当前已经有手部压力，却仍然选择抓取，则重置环境。
                print("Decision error; skip grasp and reset environment.")
                env.reset()
                env.wait(500)
                catch_success = False
                clear_runtime_cache()
                continue

            # ==================== 执行地面抓取阶段 ====================
            catch_success, catch_episode = PPO_catch_episoid(
                existing_env=env,
                episode=catch_episode,
                catch_agent=catch_agent,
                log_writer_catch=log_writer_catch,
                log_file_latest_catch=log_file_latest_catch,
                catch_checkpoint_dir=catch_checkpoint_dir,
                train_catch=train_catch,
                save_catch_checkpoint=save_catch_checkpoint,
                min_episodes_to_finish=episodes_to_finish,
            )
            continue

        # ==================== 决策分支：执行台阶上抓取 ====================
        if decision == 2:
            if expected_decision == 2:
                print(
                    "Decision correct: hand pressure detected, GPS distance is below 0.3, "
                    "and foot_y_post is above 0.03."
                )
            else:
                print("Decision error: upper-grasp conditions are not met.")

            if train_decision and log_enabled:
                correct_decisions, total_decisions = train_and_log_decision(
                    decision_agent=decision_agent,
                    log_writer_decision=log_writer_decision,
                    log_file_latest_decision=log_file_latest_decision,
                    total_episode=decision_episode,
                    decision=decision,
                    decision_reward=decision_reward,
                    decision_dict=decision_dict,
                    d_decision_state=decision_state,
                    correct_decisions=correct_decisions,
                    total_decisions=total_decisions,
                )

            if expected_decision != 2 or not train_upper:
                print("Skip upper grasp and reset environment.")
                env.reset()
                env.wait(500)
                catch_success = False
                clear_runtime_cache()
                continue
            
            left_foot_step_distance = env.left_foot_distance_to_goal()
            print(
                "Before upper-grasp training: "
                f"left-foot GPS distance to target step={left_foot_step_distance:.4f} m "
                f"(limit={UPPER_GRASP_STEP_DISTANCE_LIMIT:.4f} m)"
            )
            if (
                not np.isfinite(left_foot_step_distance)
                or left_foot_step_distance > UPPER_GRASP_STEP_DISTANCE_LIMIT
            ):
                print("Left foot is outside the target-step distance limit; reset environment.")
                env.reset()
                env.wait(500)
                catch_success = False
                clear_runtime_cache()
                continue
            
            print("Preparing upper grasp: raise head, open grippers, and settle.")
            env.raise_head_for_upper_grasp(settle_ms=600)
            env.open_grippers_fully(settle_ms=800)
            env.withdraw_left_from_step()
            env.wait(3000)

            upper_episode += 1
            print("Enter upper-grasp training stage.")
            upper_success = PPO_upper_episoid(
                existing_env=env,
                total_episode=decision_episode,
                episode=upper_episode,
                log_writer_upper=log_writer_upper,
                log_file_latest_upper=log_file_latest_upper,
                upper_agent=upper_agent,
                save_upper_checkpoint=save_upper_checkpoint,
                upper_checkpoint_dir=upper_checkpoint_dir,
                min_episodes_to_finish=episodes_to_finish,
            )
            print(f"Upper-grasp training finished, success={upper_success}")
            catch_success = False
            continue

        # ==================== 决策分支：执行踏步 ====================
        if decision == 1:
            print("Enter stepping stage.")
            print(f"Before stepping: catch_success={catch_success}")

        if decision == 1 and expected_decision == 1:
            execute_tai = True
            print("Decision correct: hand pressure detected but upper-grasp conditions are not met.")
        else:
            execute_tai = False
            print("Decision error: stepping conditions are not met.")

        if train_decision and log_enabled:
            correct_decisions, total_decisions = train_and_log_decision(
                decision_agent=decision_agent,
                log_writer_decision=log_writer_decision,
                log_file_latest_decision=log_file_latest_decision,
                total_episode=decision_episode,
                decision=decision,
                decision_reward=decision_reward,
                decision_dict=decision_dict,
                d_decision_state=decision_state,
                correct_decisions=correct_decisions,
                total_decisions=total_decisions,
            )
        elif train_decision:
            print(f"Decision training exceeded {episodes_to_finish} episodes; inference only.")
        else:
            print("Decision training disabled; inference only.")

        if not execute_tai:
            # 踏步条件不满足时，直接重置环境并清空运行缓存。
            print("Skip stepping and reset environment.")
            env.reset()
            env.wait(500)
            catch_success = False
            clear_runtime_cache()
            continue

        # ==================== 执行踏步阶段 ====================
        tai_episode += 1
        print("Stepping stage: training mode" if train_tai else "Stepping stage: inference mode")
        tai_success = PPO_tai_episoid(
            existing_env=env,
            total_episode=decision_episode,
            episode=tai_episode,
            log_writer_tai=log_writer_tai,
            log_file_latest_tai=log_file_latest_tai,
            tai_agent=tai_agent,
            keep_pose_on_success=True,
            train_tai=train_tai,
            save_tai_checkpoint=save_tai_checkpoint,
            tai_checkpoint_dir=tai_checkpoint_dir,
            min_episodes_to_finish=episodes_to_finish,
        )

        if not tai_success:
            catch_success = False
            continue

        if not train_upper:
            # 上肢训练未开启时，踏步成功后直接进入下一轮。
            print("Stepping succeeded; upper training disabled, reset environment.")
            env.reset()
            env.wait(500)
            catch_success = False
            continue

        # ==================== 复现右腿踏步 ====================
        left_step_history = getattr(env, "_last_tai_left_step_history", None)
        print("Left step succeeded; replaying right-leg step.")
        env.reset_right_leg_pose_like_env_reset()
        right_step_success = env.replay_right_step_from_left_history(left_step_history)
        print(f"Right-leg replay finished, success={right_step_success}")

        if not right_step_success:
            print("Right foot did not reach the step; skip upper grasp and reset.")
            env.reset()
            env.wait(500)
            catch_success = False
            continue

        # ==================== 上肢抓取前的姿态整理与距离检查 ====================
        print("Both feet stepped successfully; settling before upper grasp.")
        env.straighten_legs_and_settle(settle_ms=1800)
        print("Stepping finished; return to the high-level agent for the next-stage decision.")
        continue

    return 
