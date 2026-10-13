"""RobotRun3 controller for the upper stair grasp stage."""

import math


class RobotRun3:
    """Execute one upper-stage arm grasp action."""

    TIME_STEP = 32
    TARGET_X = 0.268
    TARGET_Y = 0.248
    TARGET_DIST_THRESH = 0.035
    TARGET_TOLERANCE = 0.010
    TARGET_TIMEOUT_STEPS = 220
    TARGET_STABLE_NEED = 4
    AUTO_CLOSE_MS = 640
    GOAL_REWARD = 8.0
    DISTANCE_REWARD_SCALE = 2.0
    IMU_PENALTY = 2.0
    ARM_COLLISION_PENALTY = 2.0
    LEG_COLLISION_PENALTY = 2.0
    ARM_DELTA_LIMIT = 0.3
    MAX_STEP = 24

    MOTOR_NAMES = (
        "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
        "ArmLowerR", "ArmLowerL", "PelvYR", "PelvYL", "PelvR",
        "PelvL", "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL",
        "AnkleR", "AnkleL", "FootR", "FootL", "Neck", "Head", "GraspL", "GraspR",
    )
    LIMIT = [
        [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.30], [-2.25, 0.77],
        [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.50],
        [-1.01, 1.01], [-1.00, 0.93], [-1.77, 0.45], [-0.50, 1.68],
        [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22],
        [-0.68, 1.04], [-1.02, 0.60], [-1.81, 1.81], [-0.36, 0.94],
    ]
    ACC_LOW = [470, 430, 560]
    ACC_HIGH = [570, 540, 720]
    GYRO_LOW = [495, 495, 495]
    GYRO_HIGH = [525, 525, 525]
    GRASP_TOUCH_DEVICE_NAMES = (
        "touch_grasp_L1",
        "touch_grasp_L1_1",
        "touch_grasp_L1_2",
        "touch_grasp_R1",
        "touch_grasp_R1_1",
        "touch_grasp_R1_2",
    )
    ARM_TOUCH_DEVICE_NAMES = ("touch_arm_L1", "touch_arm_R1")
    LEG_TOUCH_DEVICE_NAMES = ("touch_leg_L1", "touch_leg_L2", "touch_leg_R1", "touch_leg_R2")

    def __init__(self, robot, state, action_shoulder, action_arm_lower, step, side="left"):
        self.robot = robot
        self.timestep = self.TIME_STEP
        self.robot_state = state
        self.step = step
        self.action_shoulder = float(action_shoulder)
        self.action_arm_lower = float(action_arm_lower)
        self.side = str(side).lower()

        self.motors = []
        self.motors_sensors = []
        self._init_motors()
        self._init_sensors()

    def run(self):
        self.robot.step(self.timestep)
        imu_ok = self._within_imu_range()

        target_positions = self._build_target_positions()
        self._apply_targets(target_positions)
        reached = self._wait_targets_reached(
            target_positions={
                1: target_positions[1],
                0: target_positions[0],
                5: target_positions[5],
                4: target_positions[4],
                21: target_positions[21],
            },
            tolerance=self.TARGET_TOLERANCE,
            timeout_steps=self.TARGET_TIMEOUT_STEPS,
            stable_need=self.TARGET_STABLE_NEED,
        )
        self._post_settle(reached)

        _, left_any, right_any = self._read_grasp_contact()
        target_dist = self._target_distance()
        active_any = right_any if self.side == "right" else left_any

        if self._should_auto_close(target_dist, active_any):
            _, left_any, right_any = self._auto_close_active_gripper()
            active_any = right_any if self.side == "right" else left_any

        next_state = self._read_next_state()
        arm_collision = self._has_collision(self.touch_arm)
        leg_collision = self._has_collision(self.touch_leg)
        goal = int(active_any and target_dist <= self.TARGET_DIST_THRESH)
        reward = self._calculate_reward(goal, target_dist, imu_ok, arm_collision, leg_collision)
        done = self._is_done(goal)

        return next_state, reward, done, 1, goal, 1

    def _init_motors(self):
        for name in self.MOTOR_NAMES:
            motor = self.robot.getDevice(name)
            self.motors.append(motor)
            sensor = self.robot.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors_sensors.append(sensor)

    def _init_sensors(self):
        self.accelerometer = self.robot.getDevice("Accelerometer")
        self.gyro = self.robot.getDevice("Gyro")
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)

        self.left_gps1 = self.robot.getDevice("left_gps1")
        self.left_gps1.enable(self.timestep)
        self.right_gps1 = self.robot.getDevice("right_gps1")
        self.right_gps1.enable(self.timestep)

        self.touch_grasp = self._init_touch_sensors(self.GRASP_TOUCH_DEVICE_NAMES)
        self.touch_arm = self._init_touch_sensors(self.ARM_TOUCH_DEVICE_NAMES)
        self.touch_leg = self._init_touch_sensors(self.LEG_TOUCH_DEVICE_NAMES)

    def _init_touch_sensors(self, device_names):
        sensors = []
        for name in device_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            sensors.append(sensor)
        return sensors

    def _build_target_positions(self):
        current = self._current_key_positions()
        left_shoulder_target = 0.2995 * self.action_shoulder - 0.145
        left_lower_target = 1.25 * self.action_arm_lower + 0.25

        left_shoulder_delta = self._clamp(
            left_shoulder_target - float(self.robot_state[1]),
            -self.ARM_DELTA_LIMIT,
            self.ARM_DELTA_LIMIT,
        )
        left_arm_delta = self._clamp(
            left_lower_target - float(self.robot_state[5]),
            -self.ARM_DELTA_LIMIT,
            self.ARM_DELTA_LIMIT,
        )

        if self.side == "right":
            right_shoulder_target = self._clamp(-left_shoulder_target, self.LIMIT[0][0], self.LIMIT[0][1])
            right_lower_target = self._clamp(-left_lower_target, self.LIMIT[4][0], self.LIMIT[4][1])
            right_shoulder_delta = self._clamp(
                right_shoulder_target - float(self.robot_state[0]),
                -self.ARM_DELTA_LIMIT,
                self.ARM_DELTA_LIMIT,
            )
            right_arm_delta = self._clamp(
                right_lower_target - float(self.robot_state[4]),
                -self.ARM_DELTA_LIMIT,
                self.ARM_DELTA_LIMIT,
            )
            return {
                1: current[1],
                0: self._clamp(float(self.robot_state[0]) + right_shoulder_delta, self.LIMIT[0][0], self.LIMIT[0][1]),
                3: current[3],
                2: current[2],
                5: current[5],
                4: self._clamp(float(self.robot_state[4]) + right_arm_delta, self.LIMIT[4][0], self.LIMIT[4][1]),
                20: self._clamp(min(current[20], -0.50), -1.10, 1.10),
                21: self._clamp(max(current[21], 1.00), -1.10, 1.10),
            }

        return {
            1: self._clamp(float(self.robot_state[1]) + left_shoulder_delta, self.LIMIT[1][0], self.LIMIT[1][1]),
            0: current[0],
            3: current[3],
            2: current[2],
            5: self._clamp(float(self.robot_state[5]) + left_arm_delta, self.LIMIT[5][0], self.LIMIT[5][1]),
            4: current[4],
            20: self._clamp(max(current[20], 1.00), -1.10, 1.10),
            21: self._clamp(min(current[21], -0.50), -1.10, 1.10),
        }

    def _current_key_positions(self):
        return {
            0: float(self.motors_sensors[0].getValue()),
            1: float(self.motors_sensors[1].getValue()),
            2: float(self.motors_sensors[2].getValue()),
            3: float(self.motors_sensors[3].getValue()),
            4: float(self.motors_sensors[4].getValue()),
            5: float(self.motors_sensors[5].getValue()),
            20: float(self.motors_sensors[20].getValue()),
            21: float(self.motors_sensors[21].getValue()),
        }

    def _apply_targets(self, target_positions):
        for idx, target in target_positions.items():
            self.motors[idx].setPosition(target)

    def _wait_targets_reached(self, target_positions, tolerance, timeout_steps, stable_need):
        stable_count = 0
        key_indices = list(target_positions.keys())

        for _ in range(int(timeout_steps)):
            if self.robot.step(self.timestep) == -1:
                break

            all_reached = True
            for idx in key_indices:
                current = float(self.motors_sensors[idx].getValue())
                target = float(target_positions[idx])
                if abs(current - target) > float(tolerance):
                    all_reached = False
                    break

            if all_reached:
                stable_count += 1
                if stable_count >= int(stable_need):
                    return True
            else:
                stable_count = 0

        return False

    def _post_settle(self, reached):
        settle_steps = 10 if reached else 16
        for _ in range(settle_steps):
            if self.robot.step(self.timestep) == -1:
                break

    def _read_next_state(self):
        return [float(self.motors_sensors[i].getValue()) for i in range(20)]

    def _read_grasp_contact(self):
        values = [float(sensor.getValue()) for sensor in self.touch_grasp]
        left_any = any(value > 0.5 for value in values[:3])
        right_any = any(value > 0.5 for value in values[3:])
        return values, left_any, right_any

    def _target_distance(self):
        gps = self.right_gps1.getValues() if self.side == "right" else self.left_gps1.getValues()
        gps_x = float(gps[1]) if len(gps) > 1 else 0.0
        gps_y = float(gps[2]) if len(gps) > 2 else 0.0
        dx = self.TARGET_X - gps_x
        dy = self.TARGET_Y - gps_y
        return math.sqrt(dx * dx + dy * dy)

    def _should_auto_close(self, target_dist, contact_any):
        return target_dist <= self.TARGET_DIST_THRESH and contact_any

    def _auto_close_active_gripper(self):
        gripper_idx = 21 if self.side == "right" else 20
        self.motors[gripper_idx].setPosition(-0.5)
        close_timer = 0
        while self.robot.step(self.timestep) != -1:
            close_timer += self.timestep
            if close_timer >= self.AUTO_CLOSE_MS:
                break
        return self._read_grasp_contact()

    def _has_collision(self, sensors):
        return any(float(sensor.getValue()) > 0.5 for sensor in sensors)

    def _calculate_reward(self, goal, target_dist, imu_ok, arm_collision, leg_collision):
        reward = 0.0
        if goal == 1:
            reward += self.GOAL_REWARD
        reward -= self.DISTANCE_REWARD_SCALE * target_dist
        if not imu_ok:
            reward -= self.IMU_PENALTY
        if arm_collision:
            reward -= self.ARM_COLLISION_PENALTY
        if leg_collision:
            reward -= self.LEG_COLLISION_PENALTY
        return reward

    def _is_done(self, goal):
        return int(goal == 1 or self.step >= self.MAX_STEP)

    def _within_imu_range(self):
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            if not (self.ACC_LOW[i] <= acc[i] <= self.ACC_HIGH[i]):
                return False
            if not (self.GYRO_LOW[i] <= gyro[i] <= self.GYRO_HIGH[i]):
                return False
        return True

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))
