from core.rewards0 import Reward
from jax import random
import time
from mujoco import mjx
import mujoco
import jax
import jax.numpy as jnp

class WalkEnv():
    def __init__(self, batch_size=1):
        #恒定参数
        self.rpy=None
        self.sim_dt = 0.001  # pd控制周期
        self.control_dt = 0.005  # 决策周期,必须是sim_dt的整数倍；在一个self.control_dt内只执行一次policy，即输出的电机目标位置不变。
        model = mujoco.MjModel.from_xml_path('unitree_g1/scene_mjx.xml')
        '''
        for i in range(model.nbody):
            print(i, model.body(i).name)
        '''
        self.root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
        self.l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.l_foot_touch_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'left_foot_touch')
        self.r_foot_touch_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'right_foot_touch')
        self.l_foot_touch_adr = int(model.sensor_adr[self.l_foot_touch_sensor_id])
        self.r_foot_touch_adr = int(model.sensor_adr[self.r_foot_touch_sensor_id])
        self.nq = model.nq
        self.nv = model.nv
        self.goal_h = 0.756
        self.swing_duration = 0.72
        self.stance_duration = 0.53
        self.total_duration = 2 * (self.swing_duration + self.stance_duration)
        self.motor_offset = jnp.array([-0.312, 0, 0, 0.669, -0.363, 0,
                                       -0.312, 0 ,0, 0.669, -0.363, 0,
                                       0, 0, 0.073,
                                       0.2,  0.19, 0, 1, 0, 0, 0,
                                       0.2, -0.19, 0, 1, 0, 0, 0
                                       ])
        self.init_qpos = jnp.concatenate([jnp.array([0, 0, self.goal_h, 1, 0, 0, 0]),self.motor_offset])
        self.gear_ratios = model.actuator_gear[:, 0]
        self.gravity = abs(model.opt.gravity[2])
        self.mass = mujoco.mj_getTotalmass(model)
        self.body_weight = self.mass * self.gravity
        self.double_support_foot_frc = 0.5 * self.body_weight
        self.single_support_foot_frc = self.body_weight
        self.desired_max_foot_frc = 1.2 * self.body_weight
        #print("mass\n", self.mass)
        self.frame_skip = int(self.control_dt / self.sim_dt)
        self.kp0 = jnp.array([110, 110, 110, 130, 40, 20,
                              110, 110, 110, 130, 40, 20,
                              110, 110, 110,
                              90, 90, 90, 80, 60, 60, 60,
                              90, 90, 90, 80, 60, 60, 60,]) / self.gear_ratios
        self.kd0 = 0.1 * self.kp0
        self.batch_size = batch_size
        self.act_dim = 29
        self.obs_dim = 67
        self.model = mjx.put_model(model)  # 移到gpu
        self.batched_init_data = jax.jit(jax.vmap(self.init_data, in_axes=(0,0,0)))
        self.batched_step_data = jax.jit(jax.vmap(self.step_data, in_axes=(0,0)))
      

        #构造镜像矩阵
        '''
        self.obs_mirror_metrix=jnp.zeros((self.obs_dim,self.obs_dim))
        # 欧拉角和qx,qy,qz一样中间不变两边取反
        self.obs_mirror_metrix = self.obs_mirror_metrix.at[[1,16,32],[1,16,32]].set(1)
        self.obs_mirror_metrix = self.obs_mirror_metrix.at[[0,2,15,17,30,31], [0,2,15,17,30,31]].set(-1)
        for i in range(3,9,1):
            self.obs_mirror_metrix=self.obs_mirror_metrix.at[i+6,i].set(-1)
        for i in range(9,15,1):
            self.obs_mirror_metrix=self.obs_mirror_metrix.at[i-6,i].set(-1)
        for i in range(18,24,1):
            self.obs_mirror_metrix=self.obs_mirror_metrix.at[i+6,i].set(-1)
        for i in range(24,30,1):
            self.obs_mirror_metrix=self.obs_mirror_metrix.at[i-6,i].set(-1)
        #print("self.obs_mirror_metrix\n", self.obs_mirror_metrix)
        self.act_mirror_metrix = jnp.zeros((self.act_dim, self.act_dim))
        for i in range(6):
            self.act_mirror_metrix=self.act_mirror_metrix.at[i+6,i].set(-1)
        for i in range(6,12,1):
            self.act_mirror_metrix=self.act_mirror_metrix.at[i-6,i].set(-1)
        '''
        #print("self.act_mirror_metrix\n", self.act_mirror_metrix)

        #变量
        self.data = jax.vmap(lambda _: mjx.make_data(self.model))(jnp.arange(batch_size))
        #jax.debug.print("l_velocity:  {}", self.data.cvel.shape)
        self.goal_speed = jnp.zeros((batch_size,1))
        self.min_speed=0.2
        self.max_speed=0.4
        self.clock = jnp.zeros((batch_size,1))
        self.l_foot_force = jnp.zeros((batch_size,))
        self.r_foot_force = jnp.zeros((batch_size,))
        self.l_force_z0 = jnp.zeros((batch_size,))
        self.r_force_z0 = jnp.zeros((batch_size,))
        self.reward = Reward(self)
    def update_foot_force_state(self):
        self.l_foot_force = self.data.sensordata[:, self.l_foot_touch_adr]
        self.r_foot_force = self.data.sensordata[:, self.r_foot_touch_adr]
        self.l_force_z0 = jnp.clip(self.l_foot_force / self.desired_max_foot_frc, 0.0, 1.0)
        self.r_force_z0 = jnp.clip(self.r_foot_force / self.desired_max_foot_frc, 0.0, 1.0)

        self.time_step = 0
        #print(jax.devices())

    def if_done(self):
        dones = jnp.abs(self.goal_h - self.data.qpos[:,2]) > 0.12
        #跨到半空的动作一定有样本会摔倒，于是提前done，不能训练序列太短


        return dones

    def step_data(self,data,tau):
        data = data.replace(ctrl=tau)
        return mjx.step(self.model, data)

    def step(self,action):
        desired_target = action + self.motor_offset#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # tau=None
        for _ in range(self.frame_skip):
            motor_pos = self.data.actuator_length / self.gear_ratios
            motor_vel = self.data.actuator_velocity / self.gear_ratios
            tau = self.kp0 * (desired_target - motor_pos) - self.kd0 * motor_vel
            self.data = self.batched_step_data(self.data, tau)

        #print("tau", tau)
        self.clock = (self.clock + self.control_dt) % self.total_duration
        self.time_step += 1
        self.update_foot_force_state()
        motor_pos = self.data.actuator_length / self.gear_ratios
        motor_vel = self.data.actuator_velocity / self.gear_ratios

        roll = jnp.atan2(self.data.xmat[:,self.root_id,2,1], self.data.xmat[:,self.root_id,2,2])
        pitch = jnp.asin(-self.data.xmat[:,self.root_id,2,0])
        yaw = jnp.atan2(self.data.xmat[:,self.root_id,1,0], self.data.xmat[:,self.root_id,0,0])
        self.rpy=jnp.stack([roll,pitch,yaw],axis=-1)
        motor_bias=motor_pos  - self.motor_offset#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        obs = jnp.concatenate([self.rpy, motor_bias,
                               self.data.qvel[:,3:6], motor_vel,
                               self.clock,self.clock*self.clock,
                               #jnp.sin(2 * jnp.pi * process), jnp.cos(2 * jnp.pi * process),
                               #jnp.expand_dims(self.l_force_z0,axis=-1),jnp.expand_dims(self.r_force_z0,axis=-1),
                               self.goal_speed],axis=1)
        #obs = jnp.clip(obs, -1, 1)
        dones = self.if_done()
        total_reward = self.reward.calc_reward(jnp.squeeze(self.clock), self.data, jnp.squeeze(self.goal_speed), motor_pos, motor_vel)
        #total_reward = jnp.where(dones, total_reward - 50, total_reward)
        return obs,total_reward,dones

    def init_data(self,data,pos,vel):
        data = data.replace(qpos=pos, qvel=vel)
        return mjx.forward(self.model, data)

    def reset(self):
        subkey1, subkey2, subkey3, subkey4 = random.split(random.PRNGKey(int(time.time())), 4)
        pos = self.init_qpos + random.uniform(subkey1, (self.batch_size,self.nq), minval=-0.02, maxval=0.02)
        vel = random.uniform(subkey2, (self.batch_size,self.nv), minval=-0.02, maxval=0.02)
        pos = pos.at[:,2].set(self.goal_h)
        self.data = self.batched_init_data(self.data,pos,vel)
        self.update_foot_force_state()
        self.goal_speed = random.uniform(subkey3,(self.batch_size,1), minval=self.min_speed, maxval=self.max_speed)
        self.clock = random.uniform(subkey4, (self.batch_size,1), minval=0, maxval=self.total_duration)
        #self.clock = jnp.zeros((self.batch_size,1))
        motor_pos = self.data.actuator_length / self.gear_ratios
        motor_vel = self.data.actuator_velocity / self.gear_ratios
        #process = self.clock / self.total_duration
        roll = jnp.atan2(self.data.xmat[:, self.root_id, 2, 1], self.data.xmat[:, self.root_id, 2, 2])
        pitch = jnp.asin(-self.data.xmat[:, self.root_id, 2, 0])
        yaw = jnp.atan2(self.data.xmat[:, self.root_id, 1, 0], self.data.xmat[:, self.root_id, 0, 0])
        self.rpy = jnp.stack([roll, pitch, yaw], axis=-1)
        obs = jnp.concatenate([self.rpy, motor_pos-self.motor_offset,#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
                               vel[:,3:6],motor_vel,
                               self.clock, self.clock * self.clock,
                               #jnp.sin(2 * jnp.pi * process), jnp.cos(2 * jnp.pi * process),
                               #jnp.expand_dims(self.l_force_z0, axis=-1),jnp.expand_dims(self.r_force_z0,axis=-1),
                               self.goal_speed],axis=1)
        #obs = jnp.clip(obs, -1, 1)
        self.time_step = 0
        return obs



