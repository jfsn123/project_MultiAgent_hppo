import sys
import math
from PIL import Image
import numpy as np
import torch
sys.path.append('D:/Webots/Webots2021a/Webots/projects/robots/robotis/darwin-op/libraries/python37')
from python_scripts.PPO.utils.Project_config import Darwin_config, gps_goal, gps_goal1, path_list

class Darwin:
    """Darwin机器人类
    
    功能：封装Darwin机器人的所有操作接口
    
    属性：
        motors: 电机设备列表
        sensors: 传感器设备列表
        camera: 相机设备
        gps: GPS设备组
        touch_sensors: 触碰传感器设备列表
        其他传感器设备...
    """
    def __init__(self, robot):
        # 基础设备初始化
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        # 舵机列表初始化
        self.motors = []
        self.motors_sensors = []
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR',    'PelvYL', 
                          'PelvR',     'PelvL',     'LegUpperR', 'LegUpperL', 
                          'LegLowerR', 'LegLowerL', 'AnkleR',    'AnkleL', 
                          'FootR',     'FootL',     'Neck',      'Head', 
                          'GraspL',    'GraspR')
        self._init_motors()
        # LED设备
        self.eyeLed = robot.getDevice('EyeLed')
        self.headLed = robot.getDevice('HeadLed')
        # 传感器设备
        self.camera = robot.getDevice('Camera') # 摄像头
        self.accelerometer = robot.getDevice('Accelerometer')    # 加速度传感器
        self.gyro = robot.getDevice('Gyro')    # 陀螺仪传感器
        # GPS设备组
        self.left_gps1 = robot.getDevice('left_gps1')
        self.right_gps1 = robot.getDevice('right_gps1')
        self.left_gps2 = robot.getDevice('left_gps2')
        self.right_gps2 = robot.getDevice('right_gps2')
        self.foot_gps1 = robot.getDevice('foot_gps1')
        # 启用传感器
        self.enable_sensors()
        # 初始化触碰传感器(字典格式{传感器名称:传感器数值})
        self.touch_sensors = {}
        self._init_touch_sensors() 

    def enable_sensors(self):
        """启用所有传感器"""
        self.camera.enable(self.timestep)
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.left_gps1.enable(self.timestep)
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.foot_gps1.enable(self.timestep)

    def _init_touch_sensors(self):
        """初始化触碰传感器"""
        # 定义所有触碰传感器(字典格式{传感器名称:传感器名称})
        self.sensors_config = {
            'grasp_L1'  : 'touch_grasp_L1',
            'grasp_L1_1': 'touch_grasp_L1_1',
            'grasp_L1_2': 'touch_grasp_L1_2',
            'grasp_R1'  : 'touch_grasp_R1',
            'grasp_R1_1': 'touch_grasp_R1_1',
            'grasp_R1_2': 'touch_grasp_R1_2',
            'foot_L1'   : 'touch_foot_L1',
            'foot_L2'   : 'touch_foot_L2',
            'foot_L3'   : 'touch_foot_L3',
            'foot_R1'   : 'touch_foot_R1',
            'foot_R2'   : 'touch_foot_R2',
            'arm_L1'    : 'touch_arm_L1',
            'arm_R1'    : 'touch_arm_R1',
            'leg_L1'    : 'touch_leg_L1',
            'leg_L2'    : 'touch_leg_L2',
            'leg_R1'    : 'touch_leg_R1',
            'leg_R2'    : 'touch_leg_R2'
        }
        
        # 初始化并启用每个触碰传感器
        for key, name in self.sensors_config.items():
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.touch_sensors[key] = sensor

    def wait(self, ms):
        """等待指定的毫秒数"""
        startTime = self.robot.getTime()
        s = ms / 1000
        while s + startTime >= self.robot.getTime():
            self.robot.step(self.timestep)

    def _init_motors(self):
        """初始化所有电机和位置传感器"""
        for name in self.motorName:
            motor = self.robot.getDevice(name)
            # 电机不需要enable，创建后即可使用
            self.motors.append(motor)
            # 初始化对应的位置传感器
            sensor = self.robot.getDevice(name + 'S')
            sensor.enable(self.timestep)
            self.motors_sensors.append(sensor)

    def robot_reset(self):
        """重置机器人到初始姿态"""
        a = -np.random.random() * 0.4 + 0.7
        b = -np.random.random() * 0.4 + 1.0
        
        # 设置基础姿态
        self._set_initial_pose(a, b)
        # 设置电机速度
        self._set_motors_velocity()
        # 等待稳定
        for _ in range(24):
            self.robot.step(self.timestep)
    

    def _set_initial_pose(self, a, b):
        """设置机器人初始姿态
        
        参数：
            (a,b are defined in robot_reset())
            a: 肩部角度随机值
            b: 手臂角度随机值
        """
        pose_config = {
            'GraspL': 1, 'GraspR': 1,
            'Neck': 0.2, 'Head': 0,
            'PelvYL': 0.0, 'PelvL': 0.0,
            'LegUpperL': 0.5, 'LegLowerL': -0.2,
            'AnkleL': 0, 'FootL': 0.0,
            'PelvYR': 0.0, 'PelvR': 0.0,
            'LegUpperR': -0.5, 'LegLowerR': 0.2,
            'AnkleR': 0, 'FootR': 0.0,
            'ShoulderL': a, 'ArmUpperL': 0.67, 'ArmLowerL': -b,
            'ShoulderR': -a, 'ArmUpperR': -0.67, 'ArmLowerR': b
        }
        
        for name, position in pose_config.items():
            motor_idx = self.motorName.index(name)
            self.motors[motor_idx].setPosition(position)



    def _set_motors_velocity(self, velocity=1.0):
        """设置所有电机的速度
        
        参数：
            velocity: 目标速度值(原来文件中设置的速度都是1.0)
        """
        for motor in self.motors:
            motor.setVelocity(velocity)


    def get_gps_values(self):
        """获取所有GPS数据
        
        返回：
            tuple: (left_gps1, right_gps1, left_gps2, right_gps2, foot_gps1)
        """
        return (
            self.left_gps1.getValues(),
            self.right_gps1.getValues(),
            self.left_gps2.getValues(),
            self.right_gps2.getValues(),
            self.foot_gps1.getValues()
        )
    
    def get_touch_values(self):
        """获取所有触碰传感器的值
        
        返回：
            dict: 包含所有触碰传感器的状态的字典
                键: 传感器名称 (如 'grasp_L1', 'foot_L1' 等)
                值: 传感器读数
                    - 1.0: 传感器被触碰
                    - 0.0: 传感器未被触碰
        """
        return {key: sensor.getValue() for key, sensor in self.touch_sensors.items()}

    def get_touch_sensor_value(self, sensor_name):
        """获取指定触碰传感器的值
        参数:
            sensor_name: 传感器名称，如 'grasp_L1_2'
        返回:
            float: 传感器值，1.0表示被触碰，0.0表示未被触碰
        """
        if sensor_name in self.touch_sensors:
            return self.touch_sensors[sensor_name].getValue()
        return 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def validate_and_clean_data(self, data, default_value=0.0):
        if isinstance(data, (list, tuple)):
            return [self.validate_and_clean_data(x, default_value) for x in data]
        if isinstance(data, np.ndarray):
            return np.nan_to_num(data, nan=default_value, posinf=default_value, neginf=-default_value)
        if isinstance(data, (int, float)):
            if np.isnan(data) or np.isinf(data):
                return default_value
            return data
        return data

    def safe_touch(self, name: str) -> float:
        try:
            return float(self.get_touch_sensor_value(name))
        except Exception:
            return 0.0

    def get_hand_pressure_state(self, feature_dim: int = 6):
        pressure_values = np.array(
            [
                self.safe_touch("grasp_L1"),
                self.safe_touch("grasp_L1_1"),
                self.safe_touch("grasp_L1_2"),
                self.safe_touch("grasp_R1"),
                self.safe_touch("grasp_R1_1"),
                self.safe_touch("grasp_R1_2"),
            ],
            dtype=np.float32,
        )
        state = np.zeros(feature_dim, dtype=np.float32)
        state[: len(pressure_values)] = pressure_values
        pressure_detected = bool(np.any(pressure_values > 0.0))
        return state, pressure_values, pressure_detected

    def hand_foot_gps_distance(self) -> float:
        """Return the 3-D Euclidean distance between left_gps1 and foot_gps1."""
        try:
            gps_values = self.validate_and_clean_data(self.get_gps_values())
            left_gps = np.asarray(gps_values[0], dtype=np.float32)
            foot_gps = np.asarray(gps_values[4], dtype=np.float32)
            return float(np.linalg.norm(left_gps - foot_gps))
        except Exception:
            return 0.0

    def get_foot_y_post(self) -> float:
        """Return the current Y coordinate reported by foot_gps1."""
        try:
            gps_values = self.validate_and_clean_data(self.get_gps_values())
            foot_gps = gps_values[4]
            return float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
        except Exception:
            return 0.0

    def catch_distance_to_goal(self) -> float:
        try:
            gps_values = self.get_gps_values()
            left_gps = gps_values[0]
            dy = float(gps_goal[0]) - float(left_gps[1])
            dz = float(gps_goal[1]) - float(left_gps[2])
            distance = (dy ** 2 + dz ** 2) ** 0.5
         
            return distance
        except Exception as e:
      
            import traceback
            traceback.print_exc()
            return float("inf")

    def read_ground_grasp_state(self):
        values = [
            self.safe_touch("grasp_L1"),
            self.safe_touch("grasp_L1_1"),
            self.safe_touch("grasp_L1_2"),
            self.safe_touch("grasp_R1"),
            self.safe_touch("grasp_R1_1"),
            self.safe_touch("grasp_R1_2"),
        ]
        left_any = any(values[:3])
        right_any = any(values[3:])
        return values, left_any, right_any

    def left_foot_distance_to_goal(self) -> float:
        try:
            gps_values = self.validate_and_clean_data(self.get_gps_values())
            foot_gps = gps_values[4] if len(gps_values) > 4 else [0.0, 0.0, 0.0]
            foot_y = float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
            foot_z = float(foot_gps[2]) if len(foot_gps) > 2 else 0.0
            dy = float(gps_goal1[0]) - foot_y
            dz = float(gps_goal1[1]) - foot_z
            return (dy ** 2 + dz ** 2) ** 0.5
        except Exception:
            return float("inf")

    def open_grippers_fully(self, settle_ms: int = 600):
        try:
            self.motors[20].setPosition(1.10)
            self.motors[21].setPosition(-0.5)
            self.wait(settle_ms)
        except Exception:
            pass

    def raise_head_for_upper_grasp(self, settle_ms: int = 500):
        try:
            self.motors[19].setPosition(0.45)
            self.wait(settle_ms)
        except Exception:
            pass

    def right_foot_success(self) -> bool:
        return any(value > 0.0 for value in (self.safe_touch("foot_R1"), self.safe_touch("foot_R2")))

    def wait_joint_targets_reached(
        self,
        targets: dict,
        timeout_steps: int = 120,
        tolerance: float = 0.012,
        stable_need: int = 2,
        post_settle_steps: int = 0,
        success_check=None,
    ) -> bool:
        sensors = getattr(self, "motors_sensors", None)
        if not sensors:
            for _ in range(int(timeout_steps) + max(0, int(post_settle_steps))):
                self.robot.step(self.timestep)
                if callable(success_check) and bool(success_check()):
                    return True
            return False

        stable_count = 0
        reached = False
        for _ in range(int(timeout_steps)):
            self.robot.step(self.timestep)
            if callable(success_check) and bool(success_check()):
                return True
            all_reached = True
            for joint_idx, target in targets.items():
                curr = float(sensors[int(joint_idx)].getValue())
                if abs(curr - float(target)) > float(tolerance):
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
            self.robot.step(self.timestep)
            if callable(success_check) and bool(success_check()):
                return True
        return reached

    def reset_right_leg_pose_like_env_reset(self, steps: int = 28):
        self.tai_leg_R1()
        self.tai_leg_R2()
        boost_upper = self._clamp(-1.72, Darwin_config.limit[10][0], Darwin_config.limit[10][1])
        boost_lower = self._clamp(2.23, Darwin_config.limit[12][0], Darwin_config.limit[12][1])
        boost_ankle = self._clamp(1.05, Darwin_config.limit[14][0], Darwin_config.limit[14][1])
        self.motors[10].setPosition(boost_upper)
        self.motors[12].setPosition(boost_lower)
        self.motors[14].setPosition(boost_ankle)
        self.wait_joint_targets_reached(
            targets={10: boost_upper, 12: boost_lower, 14: boost_ankle},
            timeout_steps=120,
            tolerance=0.012,
            stable_need=2,
            post_settle_steps=0,
        )
        for _ in range(int(steps)):
            self.robot.step(self.timestep)

    def straighten_legs_and_settle(self, settle_ms: int = 800):
        try:
            self.qi_li()
        except Exception:
            pass
        target_upper_l = self._clamp(1.08, Darwin_config.limit[11][0], Darwin_config.limit[11][1])
        target_lower_l = self._clamp(-0.22, Darwin_config.limit[13][0], Darwin_config.limit[13][1])
        target_ankle_l = self._clamp(-0.05, Darwin_config.limit[15][0], Darwin_config.limit[15][1])
        target_upper_r = self._clamp(-1.08, Darwin_config.limit[10][0], Darwin_config.limit[10][1])
        target_lower_r = self._clamp(0.22, Darwin_config.limit[12][0], Darwin_config.limit[12][1])
        target_ankle_r = self._clamp(0.05, Darwin_config.limit[14][0], Darwin_config.limit[14][1])
        self.motors[11].setPosition(target_upper_l)
        self.motors[13].setPosition(target_lower_l)
        self.motors[15].setPosition(target_ankle_l)
        self.motors[10].setPosition(target_upper_r)
        self.motors[12].setPosition(target_lower_r)
        self.motors[14].setPosition(target_ankle_r)
        self.wait_joint_targets_reached(
            targets={
                11: target_upper_l,
                13: target_lower_l,
                15: target_ankle_l,
                10: target_upper_r,
                12: target_lower_r,
                14: target_ankle_r,
            },
            timeout_steps=220,
            tolerance=0.012,
            stable_need=4,
            post_settle_steps=12,
        )
        self.wait(max(int(settle_ms), 1500))

    def replay_right_step_from_left_history(self, history) -> bool:
        if not history:
            return False
        for steps, (upper, lower, ankle) in enumerate(history):
            state = self.get_robot_state()
            if len(state) <= 14:
                return False
            target_upper = self._clamp(float(state[10]) - float(upper), Darwin_config.limit[10][0], Darwin_config.limit[10][1])
            target_lower = self._clamp(float(state[12]) - float(lower), Darwin_config.limit[12][0], Darwin_config.limit[12][1])
            target_ankle = self._clamp(float(state[14]) - float(ankle), Darwin_config.limit[14][0], Darwin_config.limit[14][1])

            self.motors[10].setPosition(target_upper)
            self.motors[12].setPosition(target_lower)
            self.motors[14].setPosition(target_ankle)
            self.wait_joint_targets_reached(
                targets={10: target_upper, 12: target_lower, 14: target_ankle},
                timeout_steps=120,
                tolerance=0.012,
                stable_need=2,
                post_settle_steps=4,
                success_check=self.right_foot_success,
            )
            if self.right_foot_success():
                return True
            for _ in range(4):
                self.robot.step(self.timestep)
                if self.right_foot_success():
                    return True
        return self.right_foot_success()

    def build_tai_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
        feature_dim: int = 25,
    ):
        rs = np.array(self.validate_and_clean_data(robot_state), dtype=np.float32).reshape(-1)
        if prev_robot_state is None:
            prs = rs.copy()
        else:
            prs = np.array(self.validate_and_clean_data(prev_robot_state), dtype=np.float32).reshape(-1)
            if prs.shape[0] != rs.shape[0]:
                prs = rs.copy()

        def _idx(i: int) -> float:
            return float(rs[i]) if 0 <= i < rs.shape[0] else 0.0

        def _didx(i: int) -> float:
            if 0 <= i < rs.shape[0] and 0 <= i < prs.shape[0]:
                return float(rs[i] - prs[i])
            return 0.0

        gps_values = self.validate_and_clean_data(self.get_gps_values())
        foot_y = float(gps_values[4][1]) if len(gps_values) > 4 and len(gps_values[4]) > 1 else 0.0
        foot_z = float(gps_values[4][2]) if len(gps_values) > 4 and len(gps_values[4]) > 2 else 0.0
        # 从 Supervisor 写入的文件中读取机器人初始世界坐标（X=左右, Y=高度, Z=前后）
        try:
            with open('D:\\project_MultiAgent_hppo\\python_scripts\\robot_pos.txt', 'r') as f:
                pos_str = f.read().strip()
                bx, by, bz = pos_str.split(',')
                body_x, body_y, body_z = float(bx), float(by), float(bz)
        except Exception:
            body_x, body_y, body_z = 0.0, 0.0, 0.0
        if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
            goal_y = float(gps_goal1[0])
            goal_z = float(gps_goal1[1])
        else:
            goal_y, goal_z = 0.058, 0.023
        try:
            acc = self.accelerometer.getValues()
            acc_feat = [(float(acc[0]) - 520.0) / 100.0, (float(acc[1]) - 500.0) / 100.0, (float(acc[2]) - 640.0) / 120.0]
        except Exception:
            acc_feat = [0.0, 0.0, 0.0]

        feature = [
            _idx(10), _idx(11), _idx(12), _idx(13), _idx(14), _idx(15),
            _didx(10), _didx(11), _didx(12), _didx(13), _didx(14), _didx(15),
            foot_y, foot_z, goal_y, goal_z,
            body_x, body_y, body_z,
            *acc_feat,
            float(prev_exec_actions[0]), float(prev_exec_actions[1]), float(prev_exec_actions[2]),
            float(prev_discrete_actions[0]), float(prev_discrete_actions[1]), float(prev_discrete_actions[2]),
        ]
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape[0] > int(feature_dim):
            feature = feature[:int(feature_dim)]
        elif feature.shape[0] < int(feature_dim):
            feature = np.concatenate((feature, np.zeros(int(feature_dim) - feature.shape[0], dtype=np.float32)), axis=0)
        return feature

    def build_catch_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
    ):
        rs = np.array(self.validate_and_clean_data(robot_state), dtype=np.float32).reshape(-1)
        if prev_robot_state is None:
            prs = rs.copy()
        else:
            prs = np.array(self.validate_and_clean_data(prev_robot_state), dtype=np.float32).reshape(-1)
            if prs.shape[0] != rs.shape[0]:
                prs = rs.copy()

        def _idx(i: int) -> float:
            return float(rs[i]) if 0 <= i < rs.shape[0] else 0.0

        def _didx(i: int) -> float:
            if 0 <= i < rs.shape[0] and 0 <= i < prs.shape[0]:
                return float(rs[i] - prs[i])
            return 0.0

        gps_values = self.validate_and_clean_data(self.get_gps_values())
        gripper_y = float(gps_values[0][1]) if len(gps_values) > 0 and len(gps_values[0]) > 1 else 0.0
        gripper_z = float(gps_values[0][2]) if len(gps_values) > 0 and len(gps_values[0]) > 2 else 0.0

        if isinstance(gps_goal, (list, tuple)) and len(gps_goal) >= 2:
            goal_y = float(gps_goal[0])
            goal_z = float(gps_goal[1])
        else:
            goal_y, goal_z = 0.0, 0.0

        try:
            acc = self.accelerometer.getValues()
            acc_feat = [(float(acc[0]) - 520.0) / 100.0, (float(acc[1]) - 500.0) / 100.0, (float(acc[2]) - 640.0) / 120.0]
        except Exception:
            acc_feat = [0.0, 0.0, 0.0]

        feature = [
            _idx(0), _idx(1), _idx(4), _idx(5),
            _didx(0), _didx(1), _didx(4), _didx(5),
            gripper_y, gripper_z, goal_y, goal_z,
            *acc_feat,
            float(prev_exec_actions[0]), float(prev_exec_actions[1]),
            float(prev_discrete_actions[0]), float(prev_discrete_actions[1]),
        ]
        feature = np.asarray(feature, dtype=np.float32)
        return feature

    def build_upper_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
    ):
        rs = np.array(robot_state, dtype=np.float32).reshape(-1)
        if prev_robot_state is None:
            prs = rs.copy()
        else:
            prs = np.array(prev_robot_state, dtype=np.float32).reshape(-1)
            if prs.shape[0] != rs.shape[0]:
                prs = rs.copy()

        def _idx(i: int) -> float:
            return float(rs[i]) if 0 <= i < rs.shape[0] else 0.0

        def _didx(i: int) -> float:
            if 0 <= i < rs.shape[0] and 0 <= i < prs.shape[0]:
                return float(rs[i] - prs[i])
            return 0.0

        try:
            acc = self.accelerometer.getValues()
            acc_feat = [(float(acc[0]) - 520.0) / 100.0, (float(acc[1]) - 500.0) / 100.0, (float(acc[2]) - 640.0) / 120.0]
        except Exception:
            acc_feat = [0.0, 0.0, 0.0]

        gps_values = self.validate_and_clean_data(self.get_gps_values())
        gripper_x = float(gps_values[2][1]) if len(gps_values) > 2 and len(gps_values[2]) > 1 else 0.0
        gripper_y = float(gps_values[2][2]) if len(gps_values) > 2 and len(gps_values[2]) > 2 else 0.0

        if isinstance(gps_goal, (list, tuple)) and len(gps_goal) >= 2:
            goal_x = float(gps_goal[0])
            goal_y = float(gps_goal[1])
        else:
            goal_x, goal_y = 0.0, 0.0

        feature = [
            _idx(1), _idx(5),
            _didx(1), _didx(5),
            *acc_feat,
            float(prev_exec_actions[0]), float(prev_exec_actions[1]),
            float(prev_discrete_actions[0]), float(prev_discrete_actions[1]),
            gripper_x, gripper_y, goal_x, goal_y,
        ]
        feature = np.asarray(feature, dtype=np.float32)
        return feature

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        # 设置夹爪电机位置为闭合状态
        self.motors[20].setPosition(-0.5)
        self.motors[21].setPosition(-0.5)
      
        # 等待夹爪动作完成
        for _ in range(10):
            self.robot.step(self.timestep)

    
    def check_grasp_contact(self):
        """检查抓取器的接触状态（左右各有三个抓取传感器）
        
        返回：
            dict: 包含左右两侧抓取器接触状态的字典
                键: 'left' 或 'right'
                值: bool, 是否接触
        """
        left_grasp = any([
            self.touch_sensors['grasp_L1'].getValue(),
            self.touch_sensors['grasp_L1_1'].getValue(),
            self.touch_sensors['grasp_L1_2'].getValue()
        ])
        
        right_grasp = any([
            self.touch_sensors['grasp_R1'].getValue(),
            self.touch_sensors['grasp_R1_1'].getValue(),
            self.touch_sensors['grasp_R1_2'].getValue()
        ])
        
        return {'left': left_grasp, 'right': right_grasp}    
    #这里是从touch_peng中提取的
    def check_collision(self):
        """检查是否发生碰撞（通过检查手臂和腿部的触碰传感器）
        
        返回：
            bool: 是否发生碰撞
        """
        collision_sensors = [
            'arm_L1', 'arm_R1',
            'leg_L1', 'leg_L2',
            'leg_R1', 'leg_R2'
        ]
        
        return any(self.touch_sensors[sensor].getValue() for sensor in collision_sensors)    

    def get_camera_image(self, step):
        """获取相机图像
        
        参数：
            step: 当前步数
        返回：
            img: 原始图像数组
            img_tensor: 处理后的图像张量
        """
        # 根据当前步数生成图像文件名（与调用代码保持一致：img0.png格式）
        img_name = f"img{step}.png"
        
        # 使用Webots的相机API保存图像，100表示图像质量(0-100)
        # saveImage()会将文件保存到控制器的工作目录
        self.camera.saveImage(img_name, 100)
        
        # 使用相对路径打开图像（与原代码保持一致）
        # Webots控制器的工作目录通常是控制器所在目录
        path = f'./{img_name}'
        
        try:
            # 打开保存的图像文件
            img = Image.open(path)
        except Exception as e:
            print(f"打开图像文件失败: {e}，使用空白图像")
            # 如果文件不存在，创建一个空白图像
            img = Image.new('L', (128, 128), 255)
        
        # 将图像调整为128x128像素大小
        img = img.resize((128, 128))
        # 将图像转换为灰度图像（单通道）
        img = img.convert('L')
        # 将PIL图像转换为numpy数组，并归一化到0-1范围
        img_array = np.array(img) / 255.0
        # 将numpy数组转换为PyTorch张量：
        # - unsqueeze(0)添加batch维度
        # - float()转换为浮点类型
        img_tensor = torch.tensor(img_array).unsqueeze(0).float()
        
        # 返回原始数组和处理后的张量
        return img_array, img_tensor


    def get_robot_state(self):
        """获取机器人关节状态
        
        返回：
            list: 包含所有关节位置的列表
        """
        robot_state = [sensor.getValue() for sensor in self.motors_sensors[:-2]]
        return robot_state

    def check_acceleration_and_gyro(self):
        """检查加速度和陀螺仪数据是否在正常范围内
        
        返回：
            bool: 是否在正常范围内
        """
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        #limits 是参考RobotRun2.py和RobotRun1.py的
        acc_limits = {
            'low': [480, 430, 580],
            'high': [560, 530, 700]
        }
        gyro_limits = {
            'low': [500, 500, 500],
            'high': [520, 520, 520]
        }
        
        # 检查加速度
        for i in range(3):
            if not (acc_limits['low'][i] < acc[i] < acc_limits['high'][i]):
                return False
                
        # 检查陀螺仪
        for i in range(3):
            if not (gyro_limits['low'][i] < gyro[i] < gyro_limits['high'][i]):
                return False
                
        return True

    def check_joint_limits(self, positions):
        """检查关节位置是否在限制范围内
        
        参数：
            positions: list, 关节位置列表
        返回：
            bool: 是否所有关节都在限制范围内
        """
        #limits 依然是参考RobotRun2.py和RobotRun1.py的
        limits = [
            [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], 
            [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
            [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], 
            [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
            [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], 
            [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
            [-1.81, 1.81], [-0.36, 0.94]
        ]
        
        for i in range(len(positions)):
            pos = positions[i]
            if not (limits[i][0] <= pos <= limits[i][1]):
                return False
        return True
    #===============================================以下是固定动作组函数===============================================
    def execute_timed_motion(self, motor_positions, duration, velocity=1.0):
        """执行定时动作
        参数：
            motor_positions: dict, 电机名称和目标位置的映射
            duration: int, 动作持续时间(ms)
            velocity: float, 电机速度{默认1.0}
        """
        timer = 0
        for name, position in motor_positions.items():
            motor = self.motors[self.motorName.index(name)]
            motor.setPosition(position)
            motor.setVelocity(velocity)
        
        while self.robot.step(32) != -1:
            timer += 32
            if timer >= duration:
                break

    def _set_left_leg_initpose(self):
        """设置左腿初始姿态"""
        self.execute_timed_motion({
            'LegUpperL': 1.65, 
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 1500, 1.5)  
        self.execute_timed_motion({
            'LegUpperL': 0.4, 
            'LegLowerL': -0.7,
            'AnkleL': -0.7
        }, 1500, 1.0)
        self.execute_timed_motion({
            'LegLowerL': -0.1,
            'AnkleL': -0.15
        }, 1500, 1.0)

    def tai_leg_L1(self):
        """抬起左腿第一阶段：先把左脚撤出，避免卡在台阶下沿"""
        self.execute_timed_motion({
            'LegUpperL': -0.45,
            'LegLowerL': -2.10,
            'AnkleL': -1.20
        }, 1400, 1.6)

    def tai_leg_L2(self):
        """抬起左腿第二阶段：撤脚后再抬腿到预备踩踏位"""
        self.execute_timed_motion({
            'LegUpperL': 1.65,
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 1800, 1.8)

    def tai_leg_R1(self):
        """抬起右腿第一阶段：先把右脚撤出，避免卡在台阶下沿"""
        self.execute_timed_motion({
            'LegUpperR': 0.45,
            'LegLowerR': 2.10,
            'AnkleR': 1.20
        }, 1400, 1.6)

    def tai_leg_R2(self):
        """抬起右腿第二阶段：撤脚后再抬腿到预备踩踏位"""
        self.execute_timed_motion({
            'LegUpperR': -1.65,
            'LegLowerR': 2.2,
            'AnkleR': 0.85
        }, 1800, 1.8)

    def qi_li(self):
        """起立动作"""
        self.execute_timed_motion({
            'LegLowerL': -1.1,
            'LegUpperL': 1.45,
            'LegLowerR': 1.1,
            'LegUpperR': -1.45,
            'ArmLowerL': -0.1,
            'ArmLowerR': 0.1,
            'AnkleL': -0.25,
            'AnkleR': 0.25,
            'ShoulderL': -0.05,
            'ShoulderR': 0.05,
            'GraspL': 0,
            'GraspR': 0
        }, 2000)

    def song_L(self):
        """松开左手"""
        self.execute_timed_motion({
            'GraspL': 1
        }, 1000, 2)

    def song_R(self):
        """松开右手"""
        self.execute_timed_motion({
            'GraspR': 1
        }, 1000, 2)

    def withdraw_left_from_step(self):
        """将左臂做固定收回动作，把左爪从台阶上抽出，使用固定时长和速度。"""
        self.execute_timed_motion({'ShoulderL': -0.2}, 900, 1.2)
        self.execute_timed_motion({'ArmLowerL': -0.2}, 900, 1.2)
        self.wait(300)
        self.execute_timed_motion({'ShoulderL': 0.4}, 900, 1.2)
        self.execute_timed_motion({'ArmLowerL': -1.0}, 900, 1.2)
        
        print("左臂收回动作执行完成")

    def withdraw_right_from_step(self):
        """将右臂做固定收回动作，把右爪从台阶上抽出，使用左臂动作的镜像映射。"""
        self.execute_timed_motion({'ShoulderR': 0.2}, 900, 1.2)
        self.execute_timed_motion({'ArmLowerR': 0.2}, 900, 1.2)
        self.wait(300)
        self.execute_timed_motion({'ShoulderR': -0.4}, 900, 1.2)
        self.execute_timed_motion({'ArmLowerR': 1.0}, 900, 1.2)

        print("右臂收回动作执行完成")
            

class Environment:
    """基础环境类：管理仿真环境的基本操作（只能复用同一个 Robot 实例）"""

    def __init__(self, robot):
        # ✅ 必须从外部传入同一个 robot，禁止在这里 Robot()
        self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())
        self.darwin = Darwin(self.robot)
        self.state = None
        self.done = False
        self.isSuccess = False

    def myStep(self):
        return self.robot.step(self.timestep)

    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000
        while s + startTime >= self.robot.getTime():
            self.myStep()

    def reset(self):
        """
        重置状态函数
        """  
        self.darwin.robot_reset()   # 重置机器人状态
        # 以记事本的形式传参，作为标识符
        with open(path_list['resetFlag'], 'r+') as file:
            file.write('0')
        with open(path_list['resetFlag1'], 'r+') as file:
            file.write('0')
        self.done = False
        return self.state

    
    def step(self, state, action_shoulder, action_arm, steps, catch_flag):
        from python_scripts.PPO.Robot_Run.RobotRun1 import RobotRun
        return RobotRun(self.robot, state, action_shoulder, action_arm, steps, catch_flag).run()
    
    def step2(self, state, action_leg_upper, action_leg_lower, action_ankle, steps):
        from python_scripts.PPO.Robot_Run.RobotRun2 import RobotRun2
        return RobotRun2(self.robot, state, action_leg_upper, action_leg_lower, action_ankle, steps).run()

    def step3(self, state, action_shoulder, action_arm_lower, steps, side="left"):
        from python_scripts.PPO.Robot_Run.RobotRun3 import RobotRun3
        return RobotRun3(self.robot, state, action_shoulder, action_arm_lower, steps, side=side).run()
    
    def get_robot_state(self):
        """获取机器人的关节状态，即舵机角度"""
        return self.darwin.get_robot_state()
    
    def get_img(self, step):
        return self.darwin.get_camera_image(step=step)
    
    def print_gps(self):
        return self.darwin.get_gps_values()

    def get_hand_pressure_state(self, feature_dim: int = 6):
        return self.darwin.get_hand_pressure_state(feature_dim=feature_dim)

    def hand_foot_gps_distance(self) -> float:
        return self.darwin.hand_foot_gps_distance()

    def get_foot_y_post(self) -> float:
        return self.darwin.get_foot_y_post()

    def catch_distance_to_goal(self) -> float:
        return self.darwin.catch_distance_to_goal()

    def left_foot_distance_to_goal(self) -> float:
        """获取左脚距离目标位置的距离"""
        return self.darwin.left_foot_distance_to_goal()

    def read_ground_grasp_state(self):
        return self.darwin.read_ground_grasp_state()

    def get_gyro_data(self):
        """获取陀螺仪数据"""
        return self.darwin.gyro.getValues()

    def get_touch_sensor_value(self, sensor_name):
        return self.darwin.get_touch_sensor_value(sensor_name)

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        return self.darwin.lock_grasp()

    def validate_and_clean_data(self, data, default_value=0.0):
        """清理和验证数据"""
        return self.darwin.validate_and_clean_data(data, default_value)

    def build_tai_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
        feature_dim: int = 25,
    ):
        """构建tai阶段的特征状态"""
        return self.darwin.build_tai_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=prev_exec_actions,
            prev_discrete_actions=prev_discrete_actions,
            feature_dim=feature_dim,
        )

    def build_catch_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
    ):
        """构建catch阶段的特征状态"""
        return self.darwin.build_catch_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=prev_exec_actions,
            prev_discrete_actions=prev_discrete_actions,
        )

    def build_upper_feature_state(
        self,
        robot_state,
        prev_robot_state,
        prev_exec_actions,
        prev_discrete_actions,
    ):
        """构建upper阶段的特征状态"""
        return self.darwin.build_upper_feature_state(
            robot_state=robot_state,
            prev_robot_state=prev_robot_state,
            prev_exec_actions=prev_exec_actions,
            prev_discrete_actions=prev_discrete_actions,
        )

    def wait_joint_targets_reached(
        self,
        targets: dict,
        timeout_steps: int = 120,
        tolerance: float = 0.012,
        stable_need: int = 2,
        post_settle_steps: int = 0,
        success_check=None,
    ) -> bool:
        """等待关节到达目标位置"""
        return self.darwin.wait_joint_targets_reached(
            targets=targets,
            timeout_steps=timeout_steps,
            tolerance=tolerance,
            stable_need=stable_need,
            post_settle_steps=post_settle_steps,
            success_check=success_check,
        )

    def reset_right_leg_pose_like_env_reset(self, steps: int = 28):
        """重置右腿姿态"""
        return self.darwin.reset_right_leg_pose_like_env_reset(steps)

    def straighten_legs_and_settle(self, settle_ms: int = 800):
        """伸直腿部并稳定"""
        return self.darwin.straighten_legs_and_settle(settle_ms)

    def replay_right_step_from_left_history(self, history) -> bool:
        """从左脚历史重放右脚动作"""
        return self.darwin.replay_right_step_from_left_history(history)

    def execute_timed_motion(self, motor_positions, duration, velocity=1.0):
        """执行定时动作"""
        return self.darwin.execute_timed_motion(motor_positions, duration, velocity)

    def tai_leg_L1(self):
        """抬起左腿第一阶段"""
        return self.darwin.tai_leg_L1()

    def tai_leg_L2(self):
        """抬起左腿第二阶段"""
        return self.darwin.tai_leg_L2()

    def tai_leg_R1(self):
        """抬起右腿第一阶段"""
        return self.darwin.tai_leg_R1()

    def tai_leg_R2(self):
        """抬起右腿第二阶段"""
        return self.darwin.tai_leg_R2()

    def qi_li(self):
        """起立动作"""
        return self.darwin.qi_li()

    def song_L(self):
        """松开左手"""
        return self.darwin.song_L()

    def song_R(self):
        """松开右手"""
        return self.darwin.song_R()

    def check_grasp_contact(self):
        """检查抓取器接触状态"""
        return self.darwin.check_grasp_contact()

    def check_collision(self):
        """检查碰撞"""
        return self.darwin.check_collision()

    def check_acceleration_and_gyro(self):
        """检查加速度和陀螺仪"""
        return self.darwin.check_acceleration_and_gyro()

    def check_joint_limits(self, positions):
        """检查关节限制"""
        return self.darwin.check_joint_limits(positions)

    def open_grippers_fully(self, settle_ms: int = 600):
        """完全打开夹爪"""
        return self.darwin.open_grippers_fully(settle_ms)

    def raise_head_for_upper_grasp(self, settle_ms: int = 500):
        """抬起头部用于上方抓取"""
        return self.darwin.raise_head_for_upper_grasp(settle_ms)

    def withdraw_left_from_step(self):
        """封装：将左臂从台阶上抽出并等待稳定。"""
        return self.darwin.withdraw_left_from_step()

    def withdraw_right_from_step(self):
        """封装：将右臂从台阶上抽出并等待稳定。"""
        return self.darwin.withdraw_right_from_step()

    def right_foot_success(self) -> bool:
        """检查右脚是否成功"""
        return self.darwin.right_foot_success()

    def get_camera_image(self, step):
        """获取相机图像"""
        return self.darwin.get_camera_image(step)

    def get_robot_state(self):
        """获取机器人的关节状态，即舵机角度"""
        return self.darwin.get_robot_state()

    def get_touch_values(self):
        """获取所有触碰传感器的值"""
        return self.darwin.get_touch_values()

    def safe_touch(self, name: str) -> float:
        """安全地获取触碰传感器值"""
        return self.darwin.safe_touch(name)

    def robot_reset(self):
        """重置机器人到初始姿态"""
        return self.darwin.robot_reset()

    def enable_sensors(self):
        """启用所有传感器"""
        return self.darwin.enable_sensors()
        
