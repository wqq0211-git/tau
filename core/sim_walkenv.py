import jax.numpy as jnp
from jax import random
import mujoco
import mujoco_viewer

class SimWalkEnv():
    def __init__(self):
        self.act_dim = 29
        self.obs_dim = 67
        self.sim_dt = 0.001  # pd控制周期,与xml一致
        self.control_dt = 0.005  # 决策周期,必须是sim_dt的整数倍；在一个self.control_dt内只执行一次policy，即输出的电机目标位置不变。
        self.goal_speed = 0
        self.clock = 0
        self.goal_h = 0.756
        self.swing_duration = 0.72
        self.stance_duration = 0.53
        self.total_duration = 2 * (self.swing_duration + self.stance_duration)
        mjcf_path = "unitree_g1/scene_mjx.xml"
        r_foot_name = 'right_ankle_roll_link'
        l_foot_name = 'left_ankle_roll_link'
        root_name = 'pelvis'
        #r_force_sensor_name = "RightFootForceSensor_fsensor"
        #l_force_sensor_name = "LeftFootForceSensor_fsensor"
        self.motor_offset = jnp.array([-0.312, 0, 0, 0.669, -0.363, 0,
                                       -0.312, 0 ,0, 0.669, -0.363, 0,
                                       0,0,0.073,
                                       0.2,  0.19, 0, 1, 0, 0, 0,
                                       0.2, -0.19, 0, 1, 0, 0, 0
                                       ])
        self.desired_target = self.motor_offset.copy()
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.r_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, r_foot_name)
        self.l_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, l_foot_name)
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        #self.r_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, r_force_sensor_name)
        #self.l_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, l_force_sensor_name)
        self.gear_ratios = self.model.actuator_gear[:, 0]
        # print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 14))
        # print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 21))
        self.frame_skip = int(self.control_dt / self.sim_dt)
        self.mass = mujoco.mj_getTotalmass(self.model)
        self.kp0 = jnp.array([110, 110, 110, 120, 40, 20,
                              110, 110, 110, 120, 40, 20,
                              110, 110, 110,
                              90, 90, 90, 80, 60, 60, 60,
                              90, 90, 90, 80, 60, 60, 60,]) / self.gear_ratios
        self.kd0 = 0.1 * self.kp0
        self.init_qvel = jnp.zeros(self.model.nv)
        self.init_qpos = jnp.concatenate([jnp.array([0, 0, self.goal_h, 1, 0, 0, 0]), self.motor_offset])
        self.time=0
        self.desired_max_foot_frc = self.mass * 9.8 * 0.5

    def init_viewer(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[0] = 1
        self.viewer._render_every_frame = True

    def get_obs(self):
        root_orient = self.data.qpos[3:7]
        root_ang_vel = self.data.qvel[3:6]
        motor_pos = self.data.actuator_length/self.gear_ratios
        motor_vel = self.data.actuator_velocity/self.gear_ratios
        #process = self.clock / self.total_duration
        # '''-jnp.minimum(self.data.sensordata[:,3*self.l_force_sensor_id+2],0)
        #r_foot_force_z = -jnp.minimum(self.data.sensordata[(self.r_force_sensor_id + 1) * 3 - 1],0)
        #l_foot_force_z = -jnp.minimum(self.data.sensordata[(self.l_force_sensor_id + 1) * 3 - 1],0)
        l_height = self.data.xpos[self.l_foot_id, 2]
        r_height = self.data.xpos[self.r_foot_id, 2]
        '''
        print("root_height: ", self.data.qpos)
        print("l_height: ", l_height)
        print("r_height: ", r_height)
        '''
        #l_force_z0 = l_foot_force_z / self.desired_max_foot_frc
        #r_force_z0 = r_foot_force_z / self.desired_max_foot_frc
        # '''
        roll = jnp.atan2(self.data.xmat[self.root_id, 7], self.data.xmat[self.root_id, 8])
        pitch = jnp.asin(-self.data.xmat[self.root_id, 6])
        yaw = jnp.atan2(self.data.xmat[self.root_id, 3], self.data.xmat[self.root_id, 0])
        rpy = jnp.stack([roll, pitch, yaw], axis=-1)
        obs = jnp.concatenate([rpy, motor_pos-self.motor_offset,
                               root_ang_vel, motor_vel,
                               jnp.array([self.clock,self.clock*self.clock]),
                               #jnp.array([jnp.sin(2 * jnp.pi * process), jnp.cos(2 * jnp.pi * process)]),
                               jnp.array([self.goal_speed])])
        obs = jnp.expand_dims(obs, axis=0)
        return obs

    def if_done(self):
        #r_force_z = self.data.sensordata[(self.r_force_sensor_id + 1) * 3 - 1]
        #l_force_z = self.data.sensordata[(self.l_force_sensor_id + 1) * 3 - 1]
        done = jnp.abs(self.goal_h - self.data.qpos[2]) > 0.08
        #print("time", self.time)
        #if self.time>40 and not done:
        #    done = ((jnp.abs(r_force_z)>470) | (jnp.abs(l_force_z)>470))
        #            #|((jnp.abs(r_force_z)<150)&(jnp.abs(l_force_z)<150)))# | contact_flag
        return done

    def step(self, action):
        # print("tau:\n", action[0]*self.gear_ratios)
        desired_target = action + self.motor_offset#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # tau=None
        for _ in range(self.frame_skip):
            motor_pos = self.data.actuator_length / self.gear_ratios
            motor_vel = self.data.actuator_velocity / self.gear_ratios
            tau = self.kp0 * (desired_target - motor_pos) - self.kd0 * motor_vel
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        self.clock = (self.clock + self.control_dt) % self.total_duration
        self.time += 1
        obs = self.get_obs()
        #print(self.clock)
        total_reward = 0
        done = self.if_done()
        info = {}
        return obs, total_reward, done, info

    def reset(self):
        c = 0.02
        subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(random.PRNGKey(0), 5)
        init_qpos = self.init_qpos + random.uniform(subkey1, shape=(self.model.nq,), minval=-c, maxval=c)
        init_qvel = self.init_qvel + random.uniform(subkey2, shape=(self.model.nv,), minval=-c, maxval=c)
        init_qpos = init_qpos.at[2].set(self.goal_h)
        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = init_qvel
        mujoco.mj_forward(self.model, self.data)
        self.goal_speed = random.uniform(subkey3, minval=0.2, maxval=0.4)
        self.clock = random.uniform(subkey4, minval=0, maxval=self.total_duration)
        obs = self.get_obs()
        self.time = 0
        return obs







'''
env = WalkEnv(4)
env.reset()
action=random.uniform(random.PRNGKey(0), shape=(env.batch_size, env.act_dim), minval=-0.3, maxval=0.3)
out=env.step(action)
#print(out)
'''