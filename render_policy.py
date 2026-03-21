import argparse
import glob
import os
import pickle
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import random

from core.convnet import ACTOR_Net


def resolve_policy_path(policy_path: str | None) -> str | None:
    if policy_path and policy_path != "auto":
        return policy_path

    candidates = glob.glob("**/best_policy.pkl", recursive=True)
    candidates = [
        path for path in candidates
        if "trained/server_runs/" not in path.replace("\\", "/")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class OfflineWalkRenderer:
    def __init__(self, width: int, height: int):
        self.sim_dt = 0.001
        self.control_dt = 0.005
        self.goal_h = 0.756
        self.swing_duration = 0.72
        self.stance_duration = 0.53
        self.total_duration = 2 * (self.swing_duration + self.stance_duration)
        self.min_speed = 0.2
        self.max_speed = 0.4
        self.clock = 0.0
        self.goal_speed = 0.0
        self.time = 0

        self.model = mujoco.MjModel.from_xml_path("unitree_g1/scene_mjx.xml")
        self.data = mujoco.MjData(self.model)
        self.width = min(width, int(self.model.vis.global_.offwidth))
        self.height = min(height, int(self.model.vis.global_.offheight))
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.gear_ratios = self.model.actuator_gear[:, 0]
        self.frame_skip = int(self.control_dt / self.sim_dt)
        self.motor_offset = np.array(
            [-0.312, 0, 0, 0.669, -0.363, 0,
             -0.312, 0, 0, 0.669, -0.363, 0,
             0, 0, 0.073,
             0.2, 0.19, 0, 1, 0, 0, 0,
             0.2, -0.19, 0, 1, 0, 0, 0],
            dtype=np.float32,
        )
        self.kp0 = np.array(
            [110, 110, 110, 130, 40, 20,
             110, 110, 110, 130, 40, 20,
             110, 110, 110,
             90, 90, 90, 80, 60, 60, 60,
             90, 90, 90, 80, 60, 60, 60],
            dtype=np.float32,
        ) / self.gear_ratios
        self.kd0 = 0.1 * self.kp0
        self.init_qvel = np.zeros(self.model.nv, dtype=np.float32)
        self.init_qpos = np.concatenate(
            [np.array([0, 0, self.goal_h, 1, 0, 0, 0], dtype=np.float32), self.motor_offset]
        )

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = 1
        self.camera.distance = self.model.stat.extent * 1.5
        self.camera.lookat[0] = 2.0
        self.camera.lookat[2] = 1.5
        self.camera.elevation = -20
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[0] = 1

    def reset(self, seed: int):
        key = random.PRNGKey(seed)
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        init_qpos = self.init_qpos + np.asarray(
            random.uniform(subkey1, shape=(self.model.nq,), minval=-0.02, maxval=0.02),
            dtype=np.float32,
        )
        init_qvel = self.init_qvel + np.asarray(
            random.uniform(subkey2, shape=(self.model.nv,), minval=-0.02, maxval=0.02),
            dtype=np.float32,
        )
        init_qpos[2] = self.goal_h
        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = init_qvel
        mujoco.mj_forward(self.model, self.data)

        self.goal_speed = float(random.uniform(subkey3, minval=self.min_speed, maxval=self.max_speed))
        self.clock = float(random.uniform(subkey4, minval=0, maxval=self.total_duration))
        self.time = 0
        return self.get_obs()

    def get_obs(self):
        motor_pos = self.data.actuator_length / self.gear_ratios
        motor_vel = self.data.actuator_velocity / self.gear_ratios
        roll = np.arctan2(self.data.xmat[self.root_id, 7], self.data.xmat[self.root_id, 8])
        pitch = np.arcsin(-self.data.xmat[self.root_id, 6])
        yaw = np.arctan2(self.data.xmat[self.root_id, 3], self.data.xmat[self.root_id, 0])
        rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        obs = np.concatenate(
            [
                rpy,
                motor_pos - self.motor_offset,
                self.data.qvel[3:6],
                motor_vel,
                np.array([self.clock, self.clock * self.clock, self.goal_speed], dtype=np.float32),
            ]
        )
        return jnp.expand_dims(jnp.asarray(obs), axis=0)

    def step(self, action):
        desired_target = np.asarray(action[0], dtype=np.float32) + self.motor_offset
        for _ in range(self.frame_skip):
            motor_pos = self.data.actuator_length / self.gear_ratios
            motor_vel = self.data.actuator_velocity / self.gear_ratios
            tau = self.kp0 * (desired_target - motor_pos) - self.kd0 * motor_vel
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        self.clock = (self.clock + self.control_dt) % self.total_duration
        self.time += 1
        done = abs(self.goal_h - self.data.qpos[2]) > 0.12
        return self.get_obs(), done

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render().copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default="auto")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    policy_path = resolve_policy_path(args.policy_path)
    if policy_path is None or not os.path.exists(policy_path):
        raise FileNotFoundError("No local best_policy.pkl found for rendering.")

    output_path = args.output or f"./artifacts/render_{Path(policy_path).stem}_seed{args.seed}.mp4"
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    poster_path = output_path.replace(".mp4", ".png")

    env = OfflineWalkRenderer(width=args.width, height=args.height)
    policy = ACTOR_Net(67, 29, 32)
    with open(policy_path, "rb") as f:
        policy_params = pickle.load(f)

    obs = env.reset(args.seed)
    frames = []
    next_render_t = 0.0
    total_steps = int(args.seconds / env.control_dt)

    for _ in range(total_steps):
        sim_t = env.time * env.control_dt
        if sim_t >= next_render_t:
            frames.append(env.render())
            next_render_t += 1.0 / args.fps
        action = policy(policy_params, obs)
        obs, done = env.step(action)
        if done:
            break

    if not frames:
        frames.append(env.render())

    imageio.imwrite(poster_path, frames[min(len(frames) // 2, len(frames) - 1)])
    with imageio.get_writer(output_path, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"policy: {os.path.abspath(policy_path)}")
    print(f"video: {output_path}")
    print(f"poster: {poster_path}")
    print(f"frames: {len(frames)}")
    print(f"duration_s: {env.time * env.control_dt:.3f}")
    print(f"final_x: {float(env.data.qpos[0]):.3f}")
    print(f"final_speed: {float(env.data.qvel[0]):.3f}")


if __name__ == "__main__":
    main()
