import argparse
import glob
import os
import time
import pickle

import jax
import jax.numpy as jnp

from core.convnet import ACTOR_Net
from core.sim_walkenv import SimWalkEnv


def resolve_policy_path(policy_path: str | None) -> str | None:
    if policy_path and policy_path != "auto":
        return policy_path

    candidates = glob.glob("trained/**/best_policy.pkl", recursive=True)
    candidates = [
        path for path in candidates
        if "trained/server_runs/" not in path.replace("\\", "/")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run(policy_path: str | None, seconds: float, pause: bool):
    env = SimWalkEnv()
    policy = ACTOR_Net(env.obs_dim, env.act_dim, 32)
    policy_params = None
    resolved_policy_path = resolve_policy_path(policy_path)
    if resolved_policy_path and os.path.exists(resolved_policy_path):
        print(f"using policy: {resolved_policy_path}")
        with open(resolved_policy_path, 'rb') as f:
            policy_params = pickle.load(f)
    else:
        print("no policy found, using zero actions")
        policy = None

    states = env.reset()
    env.viewer.render()
    if pause:
        env.viewer._paused = True
        env.viewer.render()

    done = False
    t = 0.0
    if not env.viewer._paused:
        while t < seconds and not done:
            if policy is None:
                actions = jnp.zeros((1, env.act_dim))
            else:
                actions = policy(policy_params, states)
            states, rewards, done, _ = env.step(actions)
            env.viewer.render()
            time.sleep(env.control_dt)
            t += env.control_dt
    env.viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default="auto")
    parser.add_argument("--seconds", type=float, default=9.0)
    parser.add_argument("--pause", action="store_true")
    args = parser.parse_args()
    run(args.policy_path, args.seconds, args.pause)
