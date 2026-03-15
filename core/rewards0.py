import jax.numpy as jnp
import jax
import mujoco
import numpy as np
class Reward():
    def __init__(self, env):
        self.env = env
        self.halfT = self.env.swing_duration + self.env.stance_duration
        #self.desired_max_foot_frc = robot_mass * 9.8 * 0.5
        #print("frc:", self.desired_max_foot_frc)
        self.desired_max_foot_vel = 1
        self.max_height = self.env.goal_h+0.02
        self.min_height = self.env.goal_h-0.02
        self.min_foot_height = 0.035
        self.max_foot_height = 0.1
        self.a = 4 * (self.min_height-self.max_height)/self.env.swing_duration/self.env.swing_duration
        self.b = 4 * (self.min_foot_height - self.max_foot_height) / self.env.swing_duration / self.env.swing_duration
        self.weights = jnp.array([0.1, 0.075, 0.075, 0.2, 0.2, 0.2, 0.075, 0.075])
        self.scale=2.8
        self.imitate_gait=jnp.load("trace.npy")#[:,[0,2,3,6,8,9]]


    def calc_reward(self,clock, data, goal_speed, motor_pos,motor_vel):
        imitate_reward = self.calc_imitate_reward(motor_pos,clock)
        #print("imitate_reward: ", imitate_reward[0])
        motor_vel_reward = self.calc_motor_vel_reward(motor_vel)
        #print("motor_vel_reward: ", motor_vel_reward[0])
        #ctrl_reward = self.calc_ctrl_reward(desired_ctrl)
        #print("desired_target_reward: ", desired_target_reward[0])
        body_orient_reward = self.calc_body_orient_reward()
        #print("body_orient_reward: ", body_orient_reward[0])
        #foot_orient_reward = self.calc_foot_orient_reward(data)
        #print("foot_orient_reward", foot_orient_reward[0])
        fwd_vel_reward = self.calc_fwd_vel_reward(data,goal_speed)
        #print("fwd_vel_reward", fwd_vel_reward[0])
        #foot_frc_reward = self.calc_foot_frc_reward(clock)
        #print("foot_frc_reward", foot_frc_reward[0])
        foot_fwd_vel_reward = self.calc_foot_fwd_vel_reward(data, clock, goal_speed)

        foot_height_reward = self.calc_foot_height_reward(data, clock)
        #print("foot_height_reward", foot_height_reward[0])
        body_height_reward = self.calc_body_height_reward(data, clock)
        #print("body_height_reward", body_height_reward[0]
        root_accel_reward = self.calc_root_accel_reward(data)
        rewards = jnp.stack([imitate_reward,motor_vel_reward, body_orient_reward, fwd_vel_reward, foot_height_reward,foot_fwd_vel_reward, body_height_reward,root_accel_reward],axis=-1)
        #rewards = jnp.array([motor_vel_reward, desired_target_reward, body_orient_reward, body_height_reward, pd_target_reward, fwd_vel_reward, foot_frc_reward, foot_height_reward])
        #print("rewards[0]: ", rewards[0])
        total_reward = jnp.dot(rewards, self.weights)
        #print("total_reward", total_reward.shape)
        #jax.debug.print("rewards: {}", rewards[jnp.argmax(total_reward),:])
        return total_reward
        #'''----

    def calc_imitate_reward(self,motor_pos,clock):
        #idx=(clock*self.imitate_gait.shape[0]/self.env.total_duration).astype(jnp.uint8)
        leg_imitate_error = jnp.linalg.norm(self.imitate_gait[(clock/self.env.control_dt).astype(jnp.uint8),:]-motor_pos[:,:12], axis=-1)
        arm_imitate_error = jnp.linalg.norm(jnp.expand_dims(self.env.motor_offset[-14:],0)-motor_pos[:,-14:], axis=-1)
        imitate_error=0.7*leg_imitate_error+0.3*arm_imitate_error
        #jax.debug.print("leg_imitate_error:  {}", leg_imitate_error[:3])
        #jax.debug.print("arm_imitate_error:  {}", arm_imitate_error[:3])
        return jnp.exp(-imitate_error)


    def calc_ctrl_reward(self,desired_ctrl):
        ctrl_error = jnp.linalg.norm(desired_ctrl, axis=-1)
        # print("motor_vel_error: ", 0.003*motor_vel_error)
        return jnp.exp(-ctrl_error)

    def calc_root_accel_reward(self, data):
        root_accel_error = (jnp.abs(data.qvel[:,3:6]).sum(axis=-1) + jnp.abs(data.qacc[:,0:3]).sum(axis=-1))
        #jax.debug.print("root_accel_error:  {}", root_accel_error)
        return jnp.exp(-0.0005*root_accel_error)


    def calc_motor_vel_reward(self,motor_vel):
        motor_vel_error=jnp.where(jnp.abs(motor_vel) < 1.8,0,1/self.env.act_dim)
        motor_vel_error = motor_vel_error.sum(axis=-1)
        #jax.debug.print("motor_vel_error:  {}", motor_vel_error[0])
        return jnp.exp(-motor_vel_error)
        #return jnp.exp(-self.scale*motor_vel_error)

    def calc_fwd_vel_reward(self,data,goal_speed):  # 只适用正前方行走
        # forward vel reward
        fwd_vel = data.qvel[:,0].clip(self.env.min_speed,self.env.max_speed)
        fwd_vel_error = jnp.abs(fwd_vel - goal_speed)
        #print("fwd_vel_error: ", 1.8*fwd_vel_error)
        return jnp.exp(-self.scale*fwd_vel_error/(self.env.max_speed-self.env.min_speed))


    def calc_foot_fwd_vel_reward(self,data,clock,goal_speed):
        normed_l_velocity = (jnp.linalg.norm(data.cvel[:,self.env.l_foot_id, :3],axis=-1)/ goal_speed).clip(0,1)
        normed_r_velocity = (jnp.linalg.norm(data.cvel[:, self.env.r_foot_id, :3], axis=-1) / goal_speed).clip(0, 1)
        #normed_l_velocity = (data.cvel[:, self.env.l_foot_id,0] / goal_speed).clip(0, 1)
        #normed_r_velocity = (data.cvel[:, self.env.r_foot_id,0] / goal_speed).clip(0, 1)
        normed_l_velocity0 = jnp.where((clock >= 0) & (clock <= self.env.swing_duration),  0, 1)
        normed_r_velocity0 = jnp.where((clock - self.halfT >= 0) & (clock - self.halfT <= self.env.swing_duration), 0, 1)
        #normed_l_velocity0 = jnp.where((clock >= 0) & (clock <= self.env.swing_duration), normed_l_velocity, 1)
        #normed_r_velocity0 = jnp.where((clock - self.halfT >= 0) & (clock - self.halfT <= self.env.swing_duration), normed_r_velocity, 1)
        foot_fwd_vel_error = 0.5 * jnp.abs(normed_r_velocity0 - normed_r_velocity) + 0.5 * jnp.abs(normed_l_velocity0 - normed_l_velocity)
        #jax.debug.print("normed_l_velocity:  {}", normed_l_velocity)
        return jnp.exp(-foot_fwd_vel_error)
        #'''

    def calc_foot_frc_reward(self, clock):
        l_sw_force_ref = jnp.maximum(0, 16 * clock * clock - 16 * clock * self.env.swing_duration + 1)
        r_sw_force_ref = jnp.maximum(0, 16 * (clock-self.halfT) * (clock-self.halfT) - 16 * (clock-self.halfT) * self.env.swing_duration + 1)
        l_force_z0 = jnp.where((clock >= 0) & (clock <= self.env.swing_duration), 1, l_sw_force_ref)
        r_force_z0 = jnp.where((clock - self.halfT >= 0) & (clock - self.halfT <= self.env.swing_duration), 1, r_sw_force_ref)
        frc_z_error = jnp.square(r_force_z0 - self.env.r_force_z0)+jnp.square(l_force_z0 - self.env.l_force_z0)
        frc_z_reward = jnp.exp(-0.5*frc_z_error)
        #print("l_force_x0: ", l_force_x0)
        #print("r_force_x0: ", r_force_x0)
        #print("frc_x_error: ", 0.03*frc_x_error)
        #print("frc_z_error: ", frc_z_error)
        #print("frc_z_reward: ", frc_z_reward)
        return frc_z_reward

    def calc_foot_height_reward(self, data, clock):
        #print("clock: ", clock)
        r_sw_height_ref = self.b*(clock-self.env.swing_duration/2)*(clock-self.env.swing_duration/2)+self.max_foot_height
        l_sw_height_ref = self.b*(clock - self.halfT-self.env.swing_duration/2)*(clock - self.halfT-self.env.swing_duration/2)+self.max_foot_height
        l_height=data.xpos[:,self.env.l_foot_id,2].clip(self.min_foot_height,self.max_foot_height)
        r_height=data.xpos[:,self.env.r_foot_id,2].clip(self.min_foot_height,self.max_foot_height)
        l_height0 = jnp.where((clock >= 0) & (clock <= self.env.swing_duration), self.min_foot_height, l_sw_height_ref)
        r_height0 = jnp.where((clock - self.halfT >= 0) & (clock - self.halfT <= self.env.swing_duration),self.min_foot_height,r_sw_height_ref)
        #l_height0 = jnp.where((clock >= 0) & (clock <= self.env.swing_duration), l_height, l_sw_height_ref)
        #r_height0 = jnp.where((clock - self.halfT >= 0) & (clock - self.halfT <= self.env.swing_duration), r_height, r_sw_height_ref)
        foot_height_error=0.5*jnp.abs(r_height0 - r_height)+0.5*jnp.abs(l_height0 - l_height)
        #jax.debug.print("foot_height_error:  {}", foot_height_error)
        return jnp.exp(-foot_height_error/(self.max_foot_height-self.min_foot_height))

    def calc_body_height_reward(self,data, clock):
        #print("clock1: ", clock)
        clock%=self.halfT
        body_height_error=jnp.abs(self.env.goal_h - data.qpos[:,2].clip(self.min_height,self.max_height))
        #jax.debug.print("body_height_error:  {}", body_height_error.shape)
        return jnp.exp(-self.scale*body_height_error/(self.max_height-self.min_height))

    def calc_body_orient_reward(self):
        rpy = self.env.rpy.clip(-0.23,0.23)
        rpy_error = jnp.linalg.norm(rpy, axis=-1)
        return jnp.exp(-0.5*2.5*rpy_error)

    def calc_foot_orient_reward(self,data):
        #root_quat = data.xquat[:,self.env.root_id]
        l_foot_quat = data.xquat[:,self.env.l_foot_id]
        r_foot_quat = data.xquat[:,self.env.r_foot_id]
        #print("l_foot_quat.shape",l_foot_quat.shape)
        quat0 = jnp.array([1, 0, 0, 0])
        l_foot_orient_error = jnp.linalg.norm(quat0 - l_foot_quat, axis=-1)  # 默认jnp.inner(target_quat, body_quat)一定大于0
        r_foot_orient_error = jnp.linalg.norm(quat0 - r_foot_quat, axis=-1)
        #print("foot_orient_error: ", l_foot_orient_error[0])
        foot_orient_rwd=(jnp.exp(-l_foot_orient_error)+jnp.exp(-r_foot_orient_error))/2
        #root_rwd = calcu_rwd(root_quat)
        #print("l_foot_rwd\n", l_foot_rwd.shape)
        return foot_orient_rwd
        #return root_rwd


