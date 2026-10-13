import torch

if  torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 路径配置
path_list = {
    'resetFlag': 'D:/project_MultiAgent_hppo/python_scripts/resetFlag.txt',
    'resetFlag1': 'D:/project_MultiAgent_hppo/python_scripts/resetFlag1.txt',
    'photo_path_real': 'D:/project_MultiAgent_hppo/controllers/Train_main/',
  
    #PPO
    'model_path_decision_PPO_h': 'D:/project_MultiAgent_hppo/python_scripts/PPO/checkpoint/decision',
    'model_path_catch_PPO_h': 'D:/project_MultiAgent_hppo/python_scripts/PPO/checkpoint/catch',
    'model_path_tai_PPO_h': 'D:/project_MultiAgent_hppo/python_scripts/PPO/checkpoint/tai',
    'model_path_upper_PPO_h': 'D:/project_MultiAgent_hppo/python_scripts/PPO/checkpoint/upper',
    'decision_log_path_PPO':'D:/project_MultiAgent_hppo/python_scripts/PPO/log/decision_log/',
    'catch_log_path_PPO':'D:/project_MultiAgent_hppo/python_scripts/PPO/log/catch_log/',
    'tai_log_path_PPO':'D:/project_MultiAgent_hppo/python_scripts/PPO/log/tai_log/',
    'upper_log_path_PPO':'D:/project_MultiAgent_hppo/python_scripts/PPO/log/upper_log/',
}

gps_goal = [0.203, 0.183]  # 目标位置坐标1
gps_goal1 = [0.058, 0.023]  # 目标位置坐标2
gps_goal2 = [0.270, 0.252]  # 目标位置坐标3

# 机器人运动标准参数
class Darwin_config:
    limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.30], [-2.25, 0.77], 
             [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.50], 
             [-1.01, 1.01], [-1.00, 0.93], [-1.77, 0.45], [-0.50, 1.68],
             [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], 
             [-0.68, 1.04], [-1.02, 0.60], [-1.81, 1.81], [-0.36, 0.94]]  # 限制
    touch_T = [1.0, 1.0]  # 压力传感器目标值
    touch_F = [0.0, 0.0]  # 压力传感器失败值
    acc_low = [480, 450, 580]  # 加速度传感器低值
    acc_high = [560, 530, 700]  # 加速度传感器高值
    gyro_low = [500, 500, 500]  # 陀螺仪低值
    gyro_high = [520, 520, 520]  # 陀螺仪高值





