from controller import Supervisor
import numpy as np


class SupervisorRobot:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = 32
        self.robotis_op2 = self.robot.getFromDef('robotis_op2') # 获取世界坐标系中物体的对象
        self.robotis_op2_trans = self.robotis_op2.getField('translation') # 获取世界坐标系中物体的三维位置对象
        self.robotis_op2_rotation = self.robotis_op2.getField('rotation') # 获取世界坐标系中物体的三维方向对象


    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms/1000
        while s+startTime >= self.robot.getTime():
            self.robot.step(self.timestep)

    def reset(self):
        print("++++reset++++")
        DAR = [-0.0176538, 0.332399, -0.00606099]   # 机器人三维坐标的默认值
        sui_ji = [np.random.random() * 0.15 - 0.075, 0, np.random.random() * -0.02]   # 随机化机器人在三维世界中的初始位置坐标
        for i in range(0, 3):
            DAR[i] = DAR[i] + sui_ji[i]
        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置
        self.wait(100)
        self.robotis_op2_rotation.setSFRotation([0.999989, 0.000865437, 0.00465503, 0.296348])   # 调整机器人在仿真世界中的朝向
        self.wait(100)
        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置
        self.wait(100)
        # 将初始位置写入文件，供训练/测试代码读取
        with open('D:\\project_MultiAgent_hppo\\python_scripts\\robot_pos.txt', 'w') as f:
            f.write(f"{DAR[0]},{DAR[1]},{DAR[2]}")
        
    def resetsimulation(self):
        self.robot.step(self.timestep)
        while True:
            with open('D:\\project_MultiAgent_hppo\\python_scripts\\resetFlag.txt', 'r') as file:
                flag = file.read()
                if flag == '0':
                    with open('D:\\project_MultiAgent_hppo\\python_scripts\\resetFlag.txt', 'r+') as file:
                        file.write('1')
                    self.robot.simulationResetPhysics()
                    self.reset()
            self.robot.step(self.timestep)
    
supervisorRobot = SupervisorRobot()
supervisorRobot.resetsimulation()