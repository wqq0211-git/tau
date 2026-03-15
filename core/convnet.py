import jax
import jax.numpy as jnp
from jax import random
import time

class ACTOR_Net():
    def __init__(self, obs_dim, act_dim,n):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n = n

    def init_params(self):
        key = random.PRNGKey(int(time.time()))
        # 初始化策略网络参数
        return {
            #'coefficient': random.uniform(key, (self.obs_dim, self.act_dim)),
            #'bias': jnp.zeros((1, self.act_dim))
            'coefficient1': random.uniform(key, (1, self.obs_dim, self.n)),
            'bias1': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient2': random.uniform(key, (1, self.n, self.n)),
            'bias2': jnp.zeros((1, self.obs_dim, 1)),
            #'coefficient3': random.uniform(key, (1, n, n)),
            #'bias3': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient4': random.uniform(key, (1, self.n, self.act_dim)),
            'bias4': jnp.zeros((1, self.act_dim))
        }
    def __call__(self, policy_params, obs):
        #print(obs.shape)
        #HL = jnp.matmul(obs,policy_params['coefficient']) + policy_params['bias']
        # action_mean = jax.nn.relu(HL)
        reshaped_obs=jnp.expand_dims(obs, axis=-1)
        HL1 = reshaped_obs*policy_params['coefficient1'] + policy_params['bias1']
        #print("HL1",HL1)
        HL2 = jnp.matmul(jax.nn.swish(HL1),policy_params['coefficient2'])/self.n + policy_params['bias2']
        #print("HL2", HL2)
        #HL3 = jnp.matmul(jax.nn.relu(HL2),policy_params['coefficient3']) + policy_params['bias3']
        #print("HL3", HL3)
        action_mean = jnp.matmul(jax.nn.swish(HL2),policy_params['coefficient4']).mean(axis=-2)/self.n+policy_params['bias4']
        #print("action_mean", action_mean.flatten())
        #action_mean = jnp.clip(action_mean,-1,1)
        #print(action_mean.shape)
        return action_mean

class CRITIC_Net():
    def __init__(self, obs_dim,n):
        super().__init__()
        self.obs_dim = obs_dim
        self.n = n


    def init_params(self):
        key = random.PRNGKey(int(time.time()))
        # 初始化价值网络参数
        return {
            'coefficient1': random.uniform(key, (1, self.obs_dim, self.n)),
            'bias1': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient2': random.uniform(key, (1, self.n, self.n)),
            'bias2': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient3': random.uniform(key, (1, self.n, 1)),
            'bias3': jnp.ones((1, self.obs_dim)),

        }

    def __call__(self, critic_params, obs):

        reshaped_obs = jnp.expand_dims(obs, axis=-1)
        HL1 = reshaped_obs * critic_params['coefficient1'] + critic_params['bias1']
        # print("HL1",HL1)
        HL2 = jnp.matmul(jax.nn.swish(HL1), critic_params['coefficient2'])/self.n + critic_params['bias2']
        # print("HL2", HL2)
        critic = jnp.matmul(jax.nn.swish(HL2), critic_params['coefficient3']).squeeze()/self.n + critic_params['bias3']
        #jax.debug.print("critic: {}", critic.shape)
        return jnp.mean(critic,axis=1)


