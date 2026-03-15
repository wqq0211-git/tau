import mujoco.viewer
import numpy as np
import time

fps = 200#xsens采样的帧率
leg_start_t = 180#截取段的起始帧,要求右腿先摆
arm_start_t = 180 + 258#右臂截取起始帧
swing_T = 144#截取段的摆腿帧数
stand_T = 106#截取段的站腿帧数
walk_T = swing_T + stand_T#走路周期


def load_joints():
    with open("clipped_joint.txt", "r") as f:
        joint_lines = f.readlines()
    trace_len = len(joint_lines)
    joints = np.zeros((trace_len, 29))
    for i in range(trace_len):
        joints[i, :] = list(map(float, joint_lines[i].strip().split()))[:-1]
    return joints

def sim_walk():
    clipped_joints=load_joints()
    #print(clipped_joints.shape)
    clipped_joints[:,[6,9]]=-clipped_joints[:,[6,9]]
    np.save('trace.npy', clipped_joints[:,:12])
    data = np.load("trace.npy")
    print(data.shape)
    walk_T=clipped_joints.shape[0]
    # 加载模型
    model = mujoco.MjModel.from_xml_path("unitree_g1/g1_mjx.xml")
    data = mujoco.MjData(model)
    sq_len = 5*walk_T
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(sq_len):
            data.qpos[7:] = clipped_joints[i % walk_T, :]  # mended_joints[i,:]
            # mujoco.mj_step(model, data)  # 推进一步仿真
            mujoco.mj_forward(model, data)
            root_pos = np.copy(data.xpos[0])
            root_quat = np.copy(data.xquat[0])
            time.sleep(1 / fps)
            # mujoco.mj_step(model, data)  # 推进一步仿真
            viewer.sync()  # 同步渲染


def main():
    sim_walk()#查看周期运动效果




if __name__ == '__main__':
    main()



'''
joints_max=clipped_joints.max(axis=0)
joints_min=clipped_joints.min(axis=0)
print("l_leg_min\n",joints_min[6:12])
print("l_leg_max\n",joints_max[6:12])
print("r_leg_min\n",joints_min[:6])
print("r_leg_max\n",joints_max[:6])
print("waist_min\n",joints_min[12:15])
print("waist_max\n",joints_max[12:15])
print("l_arm_min\n",joints_min[22:29])
print("l_arm_max\n",joints_max[22:29])
print("r_arm_min\n",joints_min[15:22])
print("r_arm_max\n",joints_max[15:22])
print(clipped_joints.mean(axis=0))
'''
'''
print("关节名称及其在qpos中的索引:")
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_qpos_addr = model.jnt_qposadr[i]
    dof = model.jnt_dofadr[i]
    print(f"{i:2d}: 名称 = {joint_name}, qpos起始索引 = {joint_qpos_addr}, dof = {model.jnt_dofadr[i]}")
print("nq: ",model.nq)
0: 名称 = L_HIP_P, qpos起始索引 = 0, dof = 0
 1: 名称 = L_HIP_R, qpos起始索引 = 1, dof = 1
 2: 名称 = L_HIP_Y, qpos起始索引 = 2, dof = 2
 3: 名称 = L_KNEE_P, qpos起始索引 = 3, dof = 3
 4: 名称 = L_ANKLE_P, qpos起始索引 = 4, dof = 4
 5: 名称 = L_ANKLE_R, qpos起始索引 = 5, dof = 5
 6: 名称 = R_HIP_P, qpos起始索引 = 6, dof = 6
 7: 名称 = R_HIP_R, qpos起始索引 = 7, dof = 7
 8: 名称 = R_HIP_Y, qpos起始索引 = 8, dof = 8
 9: 名称 = R_KNEE_P, qpos起始索引 = 9, dof = 9
10: 名称 = R_ANKLE_P, qpos起始索引 = 10, dof = 10
11: 名称 = R_ANKLE_R, qpos起始索引 = 11, dof = 11
12: 名称 = WAIST_Y, qpos起始索引 = 12, dof = 12
13: 名称 = WAIST_R, qpos起始索引 = 13, dof = 13
14: 名称 = WAIST_P, qpos起始索引 = 14, dof = 14
15: 名称 = L_SHOULDER_P, qpos起始索引 = 15, dof = 15
16: 名称 = L_SHOULDER_R, qpos起始索引 = 16, dof = 16
17: 名称 = L_SHOULDER_Y, qpos起始索引 = 17, dof = 17
18: 名称 = L_ELBOW_Y, qpos起始索引 = 18, dof = 18
19: 名称 = L_WRIST_P, qpos起始索引 = 19, dof = 19
20: 名称 = L_WRIST_Y, qpos起始索引 = 20, dof = 20
21: 名称 = L_WRIST_R, qpos起始索引 = 21, dof = 21
22: 名称 = R_SHOULDER_P, qpos起始索引 = 22, dof = 22
23: 名称 = R_SHOULDER_R, qpos起始索引 = 23, dof = 23
24: 名称 = R_SHOULDER_Y, qpos起始索引 = 24, dof = 24
25: 名称 = R_ELBOW_Y, qpos起始索引 = 25, dof = 25
26: 名称 = R_WRIST_P, qpos起始索引 = 26, dof = 26
27: 名称 = R_WRIST_Y, qpos起始索引 = 27, dof = 27
28: 名称 = R_WRIST_R, qpos起始索引 = 28, dof = 28
29: 名称 = NECK_Y, qpos起始索引 = 29, dof = 29
nq:  30
'''
