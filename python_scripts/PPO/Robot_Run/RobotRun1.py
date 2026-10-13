"""RobotRun controller."""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from python_scripts.PPO.utils.Project_config import Darwin_config
from python_scripts.PPO.utils.Webots_interfaces import Darwin


class RobotRun(Darwin):

    # 仿真步长、等待上限和容差都沿用训练/测试阶段的经验配置。
    TIME_STEP = 32
    MAX_TARGET_WAIT_STEPS = 100
    TARGET_TOLERANCE = 0.01
    STATE_TOLERANCE = 0.005
    GRIPPER_CLOSE_MS = 2000
    ARM_DELTA_LIMIT = 0.3

    # 抓手与碰撞传感器分组：前者用于抓取判定，后者用于碰撞保护。
    GRASP_SENSOR_NAMES = (
        'grasp_L1',
        'grasp_L1_1',
        'grasp_L1_2',
        'grasp_R1',
        'grasp_R1_1',
        'grasp_R1_2',
    )
    COLLISION_SENSOR_NAMES = (
        'arm_L1',
        'arm_R1',
        'leg_L1',
        'leg_L2',
        'leg_R1',
        'leg_R2',
    )
    # 额外抽取的两个触点，用于更严格地检查抓握是否到位。
    TOUCH_CHECK_SENSOR_NAMES = ('grasp_L1_1', 'grasp_R1_2')

    def __init__(self, robot, state, action_shoulder, action_arm, step, catch_flag):
        super().__init__(robot)
        self.step = step
        self.robot_state = state
        self.catch_flag = catch_flag

        # 将策略输出映射为左右对称的肩关节和手臂关节增量。
        shoulder_delta, arm_delta = self._compute_arm_deltas(action_shoulder, action_arm)
        self.target_positions = [
            self.robot_state[1] + shoulder_delta,
            self.robot_state[0] - shoulder_delta,
            self.robot_state[5] + arm_delta,
            self.robot_state[4] - arm_delta,
        ]

        # future_state 只更新本次会被控制的几个关节，供后续越界检查使用。
        self.future_state = list(self.robot_state)
        self.future_state[1] = self.target_positions[0]
        self.future_state[0] = self.target_positions[1]
        self.future_state[5] = self.target_positions[2]
        self.future_state[4] = self.target_positions[3]

        self.touch = [self.touch_sensors[name] for name in self.TOUCH_CHECK_SENSOR_NAMES]
        self.touch_peng = [self.touch_sensors[name] for name in self.COLLISION_SENSOR_NAMES]
        self.touch_value = [0.0, 0.0]
        self.next_state = [0] * 20
        self.return_flag_list = {'done': 0, 'good': 0, 'goal': 0, 'count': 0}

    def run(self):
        # 每次执行前先走一个仿真步，让传感器和电机状态同步到最新时刻。
        self.robot.step(self.TIME_STEP)
        self._set_flags(count=1)

        # 先做状态和 IMU 安全检查，任何越界都直接结束本次动作。
        if not self._check_future_state_limits():
            self._set_flags(count=0, done=1, good=1)
            print("设置done=1，原因：future_state超出限制")
            self._read_next_state()
            return self._make_return()

        if not self._check_imu_limits():
            self._set_flags(count=0, done=1, good=0)
            print('加速度传感器值不在限制范围内,catch_flag: 0,done:1--------->405')
            self._read_next_state()
            return self._make_return()

        if self.catch_flag == 0.0:
            self._run_before_catch()
        else:
            # 已经进入抓取阶段时，先闭合夹爪，再根据触碰传感器判断结果。
            self._close_grippers()
            self._handle_grasp_result("抓取判断(catch_flag!=0)", "部分抓取状态(catch_flag!=0)")

        self._read_next_state()
        return self._make_return()

    def _compute_arm_deltas(self, action_shoulder, action_arm):
        # 策略动作先变成绝对目标，再与当前关节位置做差得到实际增量。
        current_left_arm = self.robot_state[5]
        current_left_shoulder = self.robot_state[1]
        left_arm_target = 1.25 * action_arm + 0.25
        left_shoulder_target = 0.2995 * action_shoulder - 0.145

        arm_delta = self._clamp(left_arm_target - current_left_arm, -self.ARM_DELTA_LIMIT, self.ARM_DELTA_LIMIT)
        shoulder_delta = self._clamp(
            left_shoulder_target - current_left_shoulder,
            -self.ARM_DELTA_LIMIT,
            self.ARM_DELTA_LIMIT,
        )
        return shoulder_delta, arm_delta

    def _run_before_catch(self):
        # 抓取前阶段：先移动到位，再看是否已经有接触触发。
        self._apply_arm_targets()
        self._wait_until_targets_reached()
        self._set_flags(done=0, good=1)

        self._check_collision_sensors()
        grasp_values, left_any, right_any = self._read_grasp_sensors()
        if any(grasp_values):
            self._close_grippers(log_timeout=True)
            self._handle_grasp_result("抓取判断", "部分抓取状态")
        else:
            self._update_state_after_no_grasp()

    def _apply_arm_targets(self):
        # 关节索引与目标位置一一对应，保持左右对称控制。
        for motor_index, target in zip((1, 0, 5, 4), self.target_positions):
            self.motors[motor_index].setPosition(target)

    def _wait_until_targets_reached(self):
        # 按传感器实际值等待到位，避免只发指令但电机还没跟上的情况。
        for _ in range(self.MAX_TARGET_WAIT_STEPS):
            if self.robot.step(self.TIME_STEP) == -1:
                break

            current_positions = [self.motors_sensors[i].getValue() for i in range(20)]
            all_in_position = all(
                abs(target - current_positions[i]) <= self.TARGET_TOLERANCE
                for i, target in enumerate(self.target_positions)
            )
            if all_in_position:
                break

    def _check_collision_sensors(self):
        # 只要碰撞传感器触发，就保留当前 good 标记并结束本步。
        for sensor in self.touch_peng:
            if sensor.getValue() == 1.0:
                print(f'catch_flag={self.catch_flag}, done=1---------->480ppo')
                self._set_flags(done=0, good=1, count=0)

    def _handle_grasp_result(self, judge_label, partial_label):
        # 抓取判断分两层：严格模式看固定触点组合，宽松模式看左右两侧是否都触发。
        all_grasp_sensors, left_any, right_any = self._read_grasp_sensors()
        success, fail = self._evaluate_grasp(left_any, right_any)

        print(f"{judge_label} - 传感器值: {self.touch_value}, 所有传感器: {all_grasp_sensors}")
        print(f"左侧触发: {left_any}, 右侧触发: {right_any}, 成功: {success}, 失败: {fail}")

        if fail == 1 and self.step <= 5:
            self._set_flags(count=1, done=1, good=1)
            print("失败=1且步长小于等于5")
        elif fail == 1 and self.step > 5:
            self._set_flags(count=1, done=1, good=1)
            print("失败=1且步长大于5")
        elif success == 1:
            self._set_flags(count=1, done=1, good=1, goal=1)
            print("抓取成功")
        else:
            if left_any or right_any:
                print(f"{partial_label} - 左侧: {left_any}, 右侧: {right_any}")
            self._set_flags(count=1, done=1, good=1)

    def _read_grasp_sensors(self):
        # 保存完整触点值，方便日志输出和严格比对。
        self.touch_value = [sensor.getValue() for sensor in self.touch]
        all_grasp_sensors = [self.touch_sensors[name].getValue() for name in self.GRASP_SENSOR_NAMES]
        left_any = any(all_grasp_sensors[:3])
        right_any = any(all_grasp_sensors[3:])
        return all_grasp_sensors, left_any, right_any

    def _evaluate_grasp(self, left_any, right_any):
        # success_strict / fail_strict 保持原来的经验判定，left/right 作为兜底逻辑。
        success_strict = np.array_equal(self.touch_value, Darwin_config.touch_T)
        success_relaxed = left_any and right_any
        success = int(success_strict or success_relaxed)

        fail_strict = np.array_equal(self.touch_value, Darwin_config.touch_F)
        fail_relaxed = not left_any and not right_any
        fail = int(fail_strict and fail_relaxed)
        return success, fail

    def _close_grippers(self, log_timeout=False):
        # 闭合夹爪后继续推进若干仿真步，给接触与夹持稳定留时间。
        timer = 0
        self.motors[21].setPosition(-0.5)
        self.motors[20].setPosition(-0.5)
        while self.robot.step(self.TIME_STEP) != -1:
            timer += self.TIME_STEP
            if timer >= self.GRIPPER_CLOSE_MS:
                if log_timeout:
                    print(timer)
                    print('----------------------------->532')
                break

    def _update_state_after_no_grasp(self):
        # 没有明显抓取时，检查实际关节是否已经偏离预期；偏离则提前结束。
        self._read_next_state()
        for i in range(20):
            self.next_state[i] = self.motors_sensors[i].getValue()
            diff = self.next_state[i] - self.future_state[i]
            if -self.STATE_TOLERANCE < diff < self.STATE_TOLERANCE:
                continue

            self._set_flags(count=1, done=1, good=1)
            break

    def _check_future_state_limits(self):
        # 按关节各自的物理范围检查，避免用单一关节限位误判其它关节。
        for idx, state in enumerate(self.future_state):
            limit_idx = idx if idx < len(Darwin_config.limit) else len(Darwin_config.limit) - 1
            lower, upper = Darwin_config.limit[limit_idx]
            if not (lower <= state <= upper):
                return False
        return True

    def _read_next_state(self):
        # 返回当前 20 维电机传感器值，保证外部拿到的是最新状态。
        for i in range(20):
            self.next_state[i] = self.motors_sensors[i].getValue()

    def _check_imu_limits(self):
        # 逐轴检查加速度计和陀螺仪是否都落在经验安全范围内，避免姿态异常时继续执行。
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            acc_ok = Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i]
            gyro_ok = Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]
            if not (acc_ok and gyro_ok):
                return False
        return True

    def _set_flags(self, **flags):
        # 统一更新返回标志，避免在多个分支里重复维护同一组字段。
        self.return_flag_list.update(flags)

    def _make_return(self):
        # 统一返回下一状态和四个控制标志，保持调用方接口稳定。
        return (
            self.next_state,
            self.return_flag_list['done'],
            self.return_flag_list['good'],
            self.return_flag_list['goal'],
            self.return_flag_list['count'],
        )

    @staticmethod
    def _clamp(value, lower, upper):
        # 将数值限制在闭区间内，防止关节增量过大。
        return max(lower, min(upper, value))
