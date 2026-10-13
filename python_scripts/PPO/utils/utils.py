"""
PPO 训练工具模块：模型管理、日志创建、环境辅助函数
"""
import glob
import os
import re
from typing import Optional, Tuple, Dict

import numpy as np
import torch

from python_scripts.PPO.hppo.hppo import HPPO as decision_hppo
from python_scripts.PPO.hppo.hppo_01 import HPPO as stage_hppo
from python_scripts.PPO.utils.Project_config import gps_goal, path_list




# ==================== 模型 I/O 工具 ====================

def _safe_torch_load(ckpt_path: str):
    """安全加载PyTorch checkpoint，自动选择设备"""
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.load(ckpt_path, map_location=map_location)


def safe_save_checkpoint(save_path: str, checkpoint: dict) -> None:
    """安全保存PyTorch checkpoint"""
    try:
        torch.save(checkpoint, save_path)
    except Exception as e:
        print(f"模型保存失败: {e}")


def clear_runtime_cache(images=None) -> None:
    """清空运行时缓存"""
    if images is not None:
        images.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_float(value) -> float:
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def with_ground_goal_state(robot_state):
    """Append the ground target stair x/y coordinate to a robot-state vector."""
    state = list(robot_state)
    if isinstance(gps_goal, (list, tuple)) and len(gps_goal) >= 2:
        state.extend([float(gps_goal[0]), float(gps_goal[1])])
    else:
        state.extend([0.0, 0.0])
    return state


# ==================== 目录和日志工具 ====================

def _ensure_dir(path: str) -> None:
    """创建目录，若已存在则不报错"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _next_log_file(dir_path: str, prefix: str) -> str:
    """生成下一个日志文件路径（自动递增编号）"""
    pattern = os.path.join(dir_path, f"{prefix}_*.json")
    existing = glob.glob(pattern)
    max_n = 0
    for p in existing:
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.json$", os.path.basename(p))
        if match:
            try:
                max_n = max(max_n, int(match.group(1)))
            except Exception:
                continue
    return os.path.join(dir_path, f"{prefix}_{max_n + 1}.json")


def initialize_directories() -> Tuple[str, str, str, str]:
    """
    初始化所有模型检查点目录
    返回: (decision_dir, catch_dir, tai_dir, upper_dir)
    """
    decision_dir = path_list["model_path_decision_PPO_h"]
    catch_dir = path_list["model_path_catch_PPO_h"]
    tai_dir = path_list["model_path_tai_PPO_h"]
    upper_dir = path_list["model_path_upper_PPO_h"]
    
    for model_dir in (decision_dir, catch_dir, tai_dir, upper_dir):
        _ensure_dir(model_dir)
    
    return decision_dir, catch_dir, tai_dir, upper_dir


def initialize_log_files() -> Tuple[str, str, str, str]:
    """
    初始化所有日志文件路径（自动递增编号）
    返回: (decision_log, catch_log, tai_log, upper_log)
    """
    log_dirs = {
        "decision": path_list["decision_log_path_PPO"],
        "catch": path_list["catch_log_path_PPO"],
        "tai": path_list["tai_log_path_PPO"],
        "upper": path_list["upper_log_path_PPO"], 
    }
    
    for log_dir in log_dirs.values():
        _ensure_dir(log_dir)
    
    log_file_decision = _next_log_file(log_dirs["decision"], "decision_log")
    log_file_catch = _next_log_file(log_dirs["catch"], "catch_log")
    log_file_tai = _next_log_file(log_dirs["tai"], "tai_log")
    log_file_upper = _next_log_file(log_dirs["upper"], "upper_log")
    
    print(f"将使用新的决策日志文件: {log_file_decision}")
    print(f"将使用新的抓取日志文件: {log_file_catch}")
    print(f"将使用新的踩踏日志文件: {log_file_tai}")
    print(f"将使用新的台阶上抓取日志文件: {log_file_upper}")
    
    return log_file_decision, log_file_catch, log_file_tai, log_file_upper


# ==================== 模型检查点检测 ====================

def _latest_catch_ckpt(dir_path: str):
    """查找最新的抓取模型checkpoint"""
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(file_path: str) -> int:
        match = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    selected = max(files, key=_num)
    return selected, _num(selected)


def _latest_tai_ckpt(dir_path: str):
    """查找最新的踩踏模型checkpoint"""
    files = glob.glob(os.path.join(dir_path, "tai_agent_*_*.ckpt"))
    if not files:
        return None, 0, 0

    def _nums(file_path: str):
        match = re.search(r"tai_agent_(\d+)_(\d+)\.ckpt$", os.path.basename(file_path))
        if not match:
            return 0, 0
        try:
            return int(match.group(1)), int(match.group(2))
        except Exception:
            return 0, 0

    selected = max(files, key=_nums)
    total, ep = _nums(selected)
    return selected, total, ep


def _latest_decision_ckpt(dir_path: str):
    """查找最新的决策模型checkpoint"""
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(file_path: str) -> int:
        match = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    selected = max(files, key=_num)
    return selected, _num(selected)


def _latest_upper_ckpt(dir_path: str):
    """查找最新的台阶上抓取模型checkpoint"""
    files = glob.glob(os.path.join(dir_path, "upper_agent_*_*.ckpt"))
    if not files:
        return None, 0, 0

    def _nums(file_path: str):
        parts = os.path.basename(file_path).replace(".ckpt", "").split("_")
        try:
            return int(parts[-2]), int(parts[-1])
        except Exception:
            return 0, 0

    selected = max(files, key=_nums)
    total, ep = _nums(selected)
    return selected, total, ep


# ==================== 模型加载函数 ====================

def load_catch_model(
    catch_agent,
    catch_dir: str,
    default_episode: int = 0,
    model_path: Optional[str] = None,
) -> int:
    """加载抓取模型或从头开始"""
    episode_start = default_episode
    selected_model = model_path
    if not selected_model:
        selected_model, episode_start = _latest_catch_ckpt(catch_dir)

    if not selected_model:
        print("未找到已保存的抓取模型，从头开始训练。")
        return 0

    try:
        ckpt = _safe_torch_load(selected_model)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            catch_agent.policy.load_state_dict(ckpt["policy"])
            if "optimizer_hppo" in ckpt and catch_agent.optimizer:
                catch_agent.optimizer.load_state_dict(ckpt["optimizer_hppo"])
            episode_start = int(ckpt.get("episode", episode_start))
        else:
            catch_agent.policy.load_state_dict(ckpt)
        if model_path:
            try:
                episode_start = int(os.path.basename(model_path).split("_")[-1].split(".")[0])
            except Exception:
                pass
        print(f"抓取模型加载成功: {selected_model}, episode={episode_start}")
    except Exception as e:
        print(f"抓取模型加载失败: {e}")
        episode_start = 0
    return episode_start


def load_tai_model(
    tai_agent,
    tai_dir: str,
    default_episode: int = 1,
    model_path: Optional[str] = None,
) -> int:
    """加载踩踏模型或从头开始"""
    selected_tai = model_path
    ep = default_episode
    if not selected_tai:
        selected_tai, _, ep = _latest_tai_ckpt(tai_dir)

    if not selected_tai:
        print("未找到已保存的踩踏模型，从头开始训练。")
        return default_episode

    try:
        ckpt = _safe_torch_load(selected_tai)
        if isinstance(ckpt, dict) and "policy_tai" in ckpt:
            tai_agent.policy.load_state_dict(ckpt["policy_tai"])
            if "optimizer_tai" in ckpt and tai_agent.optimizer:
                tai_agent.optimizer.load_state_dict(ckpt["optimizer_tai"])
            try:
                ep = int(ckpt.get("episode", ep))
            except Exception:
                pass
        else:
            tai_agent.policy.load_state_dict(ckpt)
        print(f"踩踏模型加载成功: {selected_tai}, episode={ep}")
        return ep
    except Exception as e:
        print(f"踩踏模型加载失败: {e}")
        return default_episode


def load_decision_model(
    decision_agent,
    dec_dir: Optional[str],
    default_episode: int = 0,
    model_path: Optional[str] = None,
) -> int:
    """加载决策模型或从头开始"""
    selected_decision = model_path
    ep = default_episode
    if not selected_decision and dec_dir:
        selected_decision, ep = _latest_decision_ckpt(dec_dir)

    if not selected_decision:
        return 0

    try:
        ckpt = _safe_torch_load(selected_decision)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            decision_agent.policy.load_state_dict(ckpt["policy"])
            if "optimizer" in ckpt and decision_agent.optimizer:
                decision_agent.optimizer.load_state_dict(ckpt["optimizer"])
            try:
                ep = int(ckpt.get("episode", ep))
            except Exception:
                pass
        else:
            decision_agent.policy.load_state_dict(ckpt)
        if model_path:
            try:
                ep = int(os.path.basename(model_path).split("_")[-1].split(".")[0])
            except Exception:
                pass
        print(f"决策模型加载成功: {selected_decision}, episode={ep}")
        return ep
    except Exception as e:
        print(f"决策模型加载失败: {e}")
        return 0


def load_upper_model(
    upper_agent,
    upper_dir: str,
    default_episode: int = 1,
    model_path: Optional[str] = None,
) -> int:
    """加载台阶上抓取模型或从头开始"""
    selected_upper = model_path
    ep = default_episode
    if not selected_upper:
        selected_upper, _, ep = _latest_upper_ckpt(upper_dir)

    if not selected_upper:
        print("未找到已保存的台阶上抓取模型，从头开始训练。")
        return default_episode

    try:
        ckpt = _safe_torch_load(selected_upper)
        if isinstance(ckpt, dict) and "policy_upper" in ckpt:
            upper_agent.policy.load_state_dict(ckpt["policy_upper"])
            if "optimizer_upper" in ckpt and upper_agent.optimizer:
                upper_agent.optimizer.load_state_dict(ckpt["optimizer_upper"])
            try:
                ep = int(ckpt.get("episode", ep))
            except Exception:
                pass
        else:
            upper_agent.policy.load_state_dict(ckpt)
        print(f"台阶上抓取模型加载成功: {selected_upper}, episode={ep}")
        return ep
    except Exception as e:
        print(f"台阶上抓取模型加载失败: {e}")
        return default_episode


def initialize_models(
    decision_agent,
    catch_agent,
    tai_agent,
    upper_agent,
    decision_dir: str,
    catch_dir: str,
    tai_dir: str,
    upper_dir: str,
    catch_model_path: Optional[str] = None,
    decision_model_path: Optional[str] = None,
    tai_model_path: Optional[str] = None,
    upper_model_path: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    """
    统一初始化所有模型
    返回: (decision_episode, catch_episode, tai_episode, upper_episode)
    """
    decision_episode = load_decision_model(decision_agent, decision_dir, default_episode=0, model_path=decision_model_path)
    catch_episode = load_catch_model(catch_agent, catch_dir, default_episode=0, model_path=catch_model_path)
    tai_episode = load_tai_model(tai_agent, tai_dir, default_episode=0, model_path=tai_model_path)
    upper_episode = load_upper_model(upper_agent, upper_dir, default_episode=0, model_path=upper_model_path)
    
    return decision_episode, catch_episode, tai_episode, upper_episode


# ==================== Agent 工厂函数 ====================

def make_agents():
    """创建所有训练agent（抓取、踩踏、台阶、决策）"""
    decision_agent = decision_hppo(
        num_servos=1,
        state_dim=8,
        use_image_input=False,
        num_actions=3,
    )
    catch_agent = stage_hppo(
        num_servos=2,
        state_dim=19,
        use_image_input=True,
    )
    tai_agent = stage_hppo(
        num_servos=3,
        state_dim=28,
        use_image_input=True,
    )
    upper_agent = stage_hppo(
        num_servos=2,
        state_dim=15,
        use_image_input=True,
    )
    return decision_agent, catch_agent, tai_agent, upper_agent




# ==================== 训练日志函数 ====================

def train_and_log_decision(
    decision_agent,
    log_writer_decision,
    log_file_latest_decision: str,
    total_episode: int,
    decision: int,
    decision_reward: float,
    decision_dict: dict,
    d_decision_state,
    correct_decisions: int,
    total_decisions: int,
):
    """执行决策agent训练和日志记录"""
    if decision_reward > 0:
        correct_decisions += 1
    total_decisions += 1
    decision_accuracy = correct_decisions / total_decisions

    log_writer_decision.add(decision_accuracy=decision_accuracy)
    log_writer_decision.add(episode_num=total_episode)
    log_writer_decision.add(decision=decision)
    log_writer_decision.add(decision_reward=decision_reward)

    decision_agent.store_transition(
        state=(d_decision_state, d_decision_state, d_decision_state),
        action=np.array([decision], dtype=np.int64),
        reward=decision_reward,
        next_state=None,
        done=True,
        value=decision_dict["value"],
        log_prob=decision_dict["discrete_log_prob"],
    )
    decision_loss = decision_agent.learn()
    if decision_loss != 0:
        log_writer_decision.add(decision_loss=decision_loss)
    log_writer_decision.save_catch(log_file_latest_decision)
    log_writer_decision.clear()
    return correct_decisions, total_decisions
