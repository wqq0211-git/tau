import time
from core.convnet import ACTOR_Net
from core.sim_walkenv import SimWalkEnv
import pickle
import jax
from core.fnet import FR_EXP_Net
def run():
    env = SimWalkEnv()
    policy = ACTOR_Net(env.obs_dim,env.act_dim,32)#FR_EXP_Net(env.obs_dim, env.act_dim,10)
    with open("trained/best_policy.pkl", 'rb') as f:
        policy_params = pickle.load(f)
    states = env.reset()
    env.viewer.render()
    env.viewer._paused = True
    env.viewer.render()
    done = False
    t=0
    if not env.viewer._paused:
        while t < 9 and not done:
            start = time.time()
            actions = policy(policy_params, states)
            #jax.debug.print("actions: {}", actions)
            states, rewards, done, _ = env.step(actions)
            #data = mjx.get_data(env.model, env.data)
            env.viewer.render()
            #end = time.time()
            #delay_time = max(0, env.control_dt - (end - start))
            #time.sleep(delay_time)
            time.sleep(env.control_dt)
            t += env.control_dt
    env.viewer.close()


if __name__=='__main__':
    run()
