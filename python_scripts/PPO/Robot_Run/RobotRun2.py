"""RobotRun2 controller for the PPO stair-stepping stage."""
import math

from python_scripts.PPO.utils.Project_config import gps_goal1


class RobotRun2:
    """Execute one left-leg stepping action and evaluate the result."""

    # 统一仿真步长、等待时间和判定阈值，和训练/测试阶段保持一致。
    TIME_STEP = 32
    TOUCH_WAIT_MS = 2000
    ACTION_SETTLE_TIMEOUT = 100
    ACTION_SETTLE_TOLERANCE = 0.01
    SUCCESS_DIST_THRESH = 0.04
    MISS_PENALTY_BASE = 4.0
    MISS_PENALTY_SCALE = 12.0
    LEG_DELTA_LIMIT = 0.3

    # 电机与传感器按腿部阶段只关注的关节/触点分组。
    MOTOR_NAMES = (
        'ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
        'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
        'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
        'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR',
    )
    LEG_TARGET_INDICES = (11, 13, 15)
    FOOT_TOUCH_DEVICE_NAMES = ('touch_foot_L1', 'touch_foot_L2', 'touch_foot_L3')
    COLLISION_TOUCH_DEVICE_NAMES = (
        'touch_arm_L1',
        'touch_arm_R1',
        'touch_leg_L1',
        'touch_leg_L2',
        'touch_leg_R1',
        'touch_leg_R2',
    )
    LIMIT = [
        [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77],
        [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.5],
        [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
        [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22],
        [-0.68, 1.04], [-1.02, 0.6], [-1.81, 1.81], [-0.36, 0.94],
    ]
    ACC_LOW = [480, 450, 580]
    ACC_HIGH = [560, 530, 700]
    GYRO_LOW = [500, 500, 500]
    GYRO_HIGH = [520, 520, 520]

    def __init__(
        self,
        robot,
        state,
        action_leg_upper,
        action_leg_lower,
        action_ankle,
        step,
    ):
        self.robot = robot
        self.timestep = self.TIME_STEP
        self.step = step
        self.goal = gps_goal1
        self.robot_state = state

        self.motors = []
        self.motors_sensors = []
        self._init_motors()

        # 读取用于安全检查和接地判断的传感器。
        self.accelerometer = robot.getDevice('Accelerometer')
        self.gyro = robot.getDevice('Gyro')
        self.foot_gps1 = self.robot.getDevice('foot_gps1')
        self.foot_gps1.enable(self.timestep)

        self.touch = self._init_touch_sensors(self.FOOT_TOUCH_DEVICE_NAMES)
        self.touch_peng = self._init_touch_sensors(self.COLLISION_TOUCH_DEVICE_NAMES)

        # 先构造未来关节状态，再把实际会执行的腿部目标写进去。
        self.future_state = list(self.robot_state)
        self.target_positions = self._build_leg_targets(action_leg_upper, action_leg_lower, action_ankle)
        for idx, target in zip(self.LEG_TARGET_INDICES, self.target_positions):
            self.future_state[idx] = target

        print("变化动作")
        print(self.target_positions)

        self.next_state = [0] * 20
        self.touch_value = [0.0] * len(self.touch)

    def run(self):
        # 先推进一步仿真，保证当前读到的传感器值是最新状态。
        self.robot.step(self.TIME_STEP)

        goal = 0
        reward = 0
        count = 1

        # 对腿部目标做一次限幅，避免把关节推到非法范围。
        self._clamp_future_leg_state()

        # 给目标一个短暂的稳定窗口，等待关节真正跟上命令。
        for _ in range(3):
            self.robot.step(self.TIME_STEP)

        if not self._within_imu_range():
            print("传感器数据异常，返回零奖励并结束回合")
            self._read_next_state()
            return self._make_return(reward=0, done=1, good=0, goal=goal, count=0)

        # 只下发腿部关键关节的目标位置。
        final_target = [float(self.future_state[idx]) for idx in self.LEG_TARGET_INDICES]
        self._apply_leg_targets(final_target)

        # 等待目标关节到位，并要求连续稳定若干帧，减少抖动误判。
        reached, used_frames = self._wait_leg_target_reached(final_target)
        print(
            f"[SETTLE] step={self.step} reached={int(reached)} "
            f"frames={used_frames}/{self.ACTION_SETTLE_TIMEOUT} tol={self.ACTION_SETTLE_TOLERANCE:.3f}"
        )

        # 以目标点距离作为主奖励项，后续再叠加接触和碰撞分支。
        total_distance = self._distance_to_goal_after_action()
        reward = -total_distance
        done = 0
        good = 1

        if self._has_collision():
            self._read_next_state()
            return self._make_return(reward=-10.0, done=0, good=1, goal=goal, count=0)

        if self._has_foot_contact():
            print("___________")
            for sensor in self.touch:
                print(sensor.getValue())

            self._wait_after_foot_contact()
            self._read_foot_touch_values()
            return self._handle_foot_contact(total_distance)

        self._read_next_state()
        return self._make_return(reward=reward, done=done, good=good, goal=goal, count=count)

    def _init_motors(self):
        for name in self.MOTOR_NAMES:
            self.motors.append(self.robot.getDevice(name))
            sensor = self.robot.getDevice(name + 'S')
            sensor.enable(self.timestep)
            self.motors_sensors.append(sensor)

    def _init_touch_sensors(self, device_names):
        sensors = []
        for name in device_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            sensors.append(sensor)
        return sensors

    def _build_leg_targets(self, action_leg_upper, action_leg_lower, action_ankle):
        # 策略动作先变成绝对目标，再与当前关节位置做差得到实际增量。
        # 线性映射：将策略输出[-1, 1]映射到各关节的物理限制范围
        # UpperLegL (11): [-0.5, 1.68]
        # LowerLegL (13): [-2.25, 0.03]
        # AnkleL (15): [-1.39, 1.22]
        leg_upper_target = 1.09 * action_leg_upper + 0.59
        leg_lower_target = 1.14 * action_leg_lower - 1.11
        ankle_target = 1.305 * action_ankle - 0.085

        # 计算增量并限制单次移动量
        leg_upper_delta = self._clamp(leg_upper_target - self.robot_state[11], -self.LEG_DELTA_LIMIT, self.LEG_DELTA_LIMIT)
        leg_lower_delta = self._clamp(leg_lower_target - self.robot_state[13], -self.LEG_DELTA_LIMIT, self.LEG_DELTA_LIMIT)
        ankle_delta = self._clamp(ankle_target - self.robot_state[15], -self.LEG_DELTA_LIMIT, self.LEG_DELTA_LIMIT)

        return [
            self.robot_state[11] + leg_upper_delta,
            self.robot_state[13] + leg_lower_delta,
            self.robot_state[15] + ankle_delta,
        ]

    def _clamp_future_leg_state(self):
        # 把未来状态里的三个腿部关节限制在各自物理范围内。
        raw_values = {idx: self.future_state[idx] for idx in self.LEG_TARGET_INDICES}
        for idx in self.LEG_TARGET_INDICES:
            self.future_state[idx] = self._clamp_joint(idx, self.future_state[idx])

        if any(raw_values[idx] != self.future_state[idx] for idx in self.LEG_TARGET_INDICES):
            print(
                f"[CLAMP] step={self.step} "
                f"11:{raw_values[11]:.3f}->{self.future_state[11]:.3f} "
                f"13:{raw_values[13]:.3f}->{self.future_state[13]:.3f} "
                f"15:{raw_values[15]:.3f}->{self.future_state[15]:.3f}"
            )

    def _within_imu_range(self):
        # IMU 超范围通常意味着姿态异常，直接判定本步无效。
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            acc_ok = self.ACC_LOW[i] < acc[i] < self.ACC_HIGH[i]
            gyro_ok = self.GYRO_LOW[i] < gyro[i] < self.GYRO_HIGH[i]
            if not (acc_ok and gyro_ok):
                return False
        return True

    def _apply_leg_targets(self, target):
        # 只对关键腿部关节下发位置控制指令。
        for idx, value in zip(self.LEG_TARGET_INDICES, target):
            self.motors[idx].setPosition(value)

    def _wait_leg_target_reached(self, target, timeout=None, tolerance=None):
        # 通过关节传感器判断是否真正到位，而不是只看命令是否已发送。
        timeout = self.ACTION_SETTLE_TIMEOUT if timeout is None else timeout
        tolerance = self.ACTION_SETTLE_TOLERANCE if tolerance is None else tolerance

        for frame_idx in range(int(timeout)):
            if self.robot.step(self.TIME_STEP) == -1:
                return False, frame_idx + 1

            current_positions = [float(self.motors_sensors[idx].getValue()) for idx in self.LEG_TARGET_INDICES]
            if all(abs(float(target[i]) - current_positions[i]) <= tolerance for i in range(len(target))):
                return True, frame_idx + 1

        return False, int(timeout)

    def _distance_to_goal_after_action(self):
        # 用脚部 GPS 到目标点的平面距离作为动作后效果评估。
        foot_gps_now = self.foot_gps1.getValues()
        x_distance = self.goal[0] - float(foot_gps_now[1])
        y_distance = self.goal[1] - float(foot_gps_now[2])
        return math.sqrt(x_distance ** 2 + y_distance ** 2)

    def _has_collision(self):
        # 只要任一碰撞触点被触发，就认为发生了碰撞。
        return any(sensor.getValue() > 0.0 for sensor in self.touch_peng)

    def _has_foot_contact(self):
        # 判断脚部是否已经接触到目标区域或地面。
        return any(sensor.getValue() > 0.0 for sensor in self.touch)

    def _wait_after_foot_contact(self):
        # 触地后再等待一小段时间，让接触稳定下来。
        timer = 0
        while self.robot.step(self.TIME_STEP) != -1:
            timer += self.TIME_STEP
            if timer >= self.TOUCH_WAIT_MS:
                break

    def _read_foot_touch_values(self):
        # 保存触点值，供后续成功/失败分支使用。
        self.touch_value = [sensor.getValue() for sensor in self.touch]

    def _handle_foot_contact(self, total_distance):
        # 触地后按距离判断是成功踩点还是偏离目标。
        left_touch_any = any(value > 0.0 for value in self.touch_value)
        if not left_touch_any:
            self._read_next_state()
            return self._make_return(reward=-2.0, done=1, good=1, goal=0, count=1)

        if total_distance < self.SUCCESS_DIST_THRESH:
            closeness_ratio = max(0.0, (self.SUCCESS_DIST_THRESH - total_distance) / self.SUCCESS_DIST_THRESH)
            reward = 50.0 + 80.0 * closeness_ratio
            print(
                f"[STEP_OK] step={self.step} dist={total_distance:.4f} < {self.SUCCESS_DIST_THRESH:.4f}, "
                f"closeness={closeness_ratio:.3f}, reward={reward:.2f}"
            )
            self._read_next_state()
            return self._make_return(reward=reward, done=1, good=1, goal=1, count=1)

        miss_ratio = min(1.0, max(0.0, (total_distance - self.SUCCESS_DIST_THRESH) / self.SUCCESS_DIST_THRESH))
        miss_penalty = self.MISS_PENALTY_BASE + self.MISS_PENALTY_SCALE * miss_ratio
        print(
            f"[STEP_OFF_TARGET] step={self.step} dist={total_distance:.4f} >= {self.SUCCESS_DIST_THRESH:.4f}, "
            f"miss_ratio={miss_ratio:.3f}, penalty={miss_penalty:.2f}"
        )
        self._read_next_state()
        return self._make_return(reward=-miss_penalty, done=1, good=1, goal=0, count=1)

    def _read_next_state(self):
        # 返回当前 20 维电机传感器值，供外部策略继续使用。
        for i in range(20):
            self.next_state[i] = self.motors_sensors[i].getValue()

    def _make_return(self, reward, done, good, goal, count):
        # 统一封装本阶段的下一状态、奖励和终止标志。
        return self.next_state, reward, done, good, goal, count

    def _clamp(self, value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def _clamp_joint(self, idx, value):
        # 按每个关节对应的上下限裁剪动作。
        return max(self.LIMIT[idx][0], min(self.LIMIT[idx][1], value))
