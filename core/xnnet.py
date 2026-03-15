import jax
import jax.numpy as jnp
from jax import random
import time

class ACTOR_Net():
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def init_params(self,n):
        key = random.PRNGKey(int(time.time()))
        # 初始化策略网络参数
        return {
            'coefficient1': 0.5*(random.uniform(key, (1, self.obs_dim, n))- 0.5),
            'bias1': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient2': 0.5*(random.uniform(key, (1, n, n)) - 0.5),
            'bias2': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient3': 0.5*(random.uniform(key, (1, n, n)) - 0.5),
            'bias3': jnp.zeros((1, self.obs_dim, 1)),
            'coefficient4': 0.5*(random.uniform(key, (1, n, self.act_dim)) - 0.5),
            'bias4': jnp.zeros((1, self.act_dim))
        }
    def __call__(self, policy_params, obs):
        reshaped_obs=jnp.expand_dims(obs, axis=-1)
        HL1 = reshaped_obs*policy_params['coefficient1'] + policy_params['bias1']
        HL2 = jnp.matmul(jax.nn.relu(HL1),policy_params['coefficient2']) + policy_params['bias2']
        HL3 = jnp.matmul(jax.nn.relu(HL2),policy_params['coefficient3']) + policy_params['bias3']
        action_mean = jnp.matmul(jax.nn.relu(HL3),policy_params['coefficient4']).mean(axis=-2)+policy_params['bias4']
        #print(action_mean.shape)
        return action_mean

class CRITIC_Net():
    def __init__(self, obs_dim, head_len):
        self.obs_dim = obs_dim
        self.sqrt_dk = jnp.sqrt(head_len)
        self.head_len = head_len

    def init_params(self, n):
        key = random.PRNGKey(int(time.time()))
        # 初始化价值网络参数
        return {
            #'query_weights': 0.5*(random.uniform(key, (1,1, self.obs_dim))- 0.5),
            #'key_weights' : 0.5*(random.uniform(key, (1,1, self.obs_dim))- 0.5),
            #'value_weights': 0.5*(random.uniform(key, (1,1, self.obs_dim))- 0.5),
            #'query_bias':0.0, 'key_bias':0.0, 'value_bias':0.0,
            'weights1': 0.5*(random.uniform(key, (1, 1, self.obs_dim, n))- 0.5),
            'weights2': 0.5*(random.uniform(key, (1, 1, n, n))- 0.5),
            'weights3': 0.5*(random.uniform(key, (1, 1, n, n)) - 0.5),
            'weights4': 0.5*(random.uniform(key, (1, 1, n, 1))-0.5),
            #'coefficient1': random.uniform(key, (1,1, self.obs_dim)),
            #'coefficient2': random.uniform(key, (1,1, self.obs_dim, self.obs_dim)),
            #'coefficient3': random.uniform(key, (1,1, self.obs_dim, self.obs_dim, self.obs_dim)),
            'bias1': jnp.zeros((1, 1, self.obs_dim, 1)),
            'bias2': jnp.zeros((1, 1, self.obs_dim, 1)),
            'bias3': jnp.zeros((1, 1, self.obs_dim, 1)),
            'bias4': 0.6
            }
    def __call__(self, critic_params, obs):
        '''
        attention_obs = jnp.zeros_like(obs)
        query = critic_params['query_weights'] * obs + critic_params['query_bias']
        key = critic_params['key_weights'] * obs + critic_params['key_bias']
        value = critic_params['value_weights'] * obs + critic_params['value_bias']
        idx=0
        for i in range(self.head_len.shape[0]):
            query_e=query[:,:,idx:idx+self.head_len[i]]
            key_e=key[:,:,idx:idx+self.head_len[i]]
            value_e=value[:,:,idx:idx+self.head_len[i]]
            scores = jnp.matmul(jnp.swapaxes(key_e, -1, -2),query_e) / self.sqrt_dk[i]
            attention_weights = jax.nn.softmax(scores, axis=-1)
            attention_obs=attention_obs.at[:,:,idx:idx+self.head_len[i]].set(jnp.matmul(value_e,attention_weights))
            idx += self.head_len[i]
        '''
        #encoded_obs = jnp.expand_dims(attention_obs, axis=-1)
        encoded_obs = jnp.expand_dims(obs, axis=-1)
        HL1 = encoded_obs*critic_params['weights1'] + critic_params['bias1']
        HL2 = jnp.matmul(jax.nn.relu(HL1),critic_params['weights2']) + critic_params['bias2']
        HL3 = jnp.matmul(jax.nn.relu(HL2), critic_params['weights3']) + critic_params['bias3']
        critic = jnp.matmul(jax.nn.relu(HL3), critic_params['weights4']).squeeze(axis=-1).mean(axis=-1)+ critic_params['bias4']
        #print("critic", critic.shape)
        '''
        rwd1 = critic_params['coefficient1']*encoded_obs
        v1 = jnp.expand_dims(encoded_obs, axis=-1)
        v2 = jnp.expand_dims(encoded_obs, axis=-2)
        M1 = v1*v2
        rwd2 = critic_params['coefficient2']*M1
        M2 = jnp.expand_dims(M1, axis=-1)
        M3 = jnp.expand_dims(v2, axis=-2)
        rwd3 = critic_params['coefficient3']*jnp.triu(M2*M3)
        critic = rwd1.mean(axis=-1)+rwd2.mean(axis=(-2,-1))+rwd3.mean(axis=(-3,-2,-1))+critic_params['bias']
        '''
        return critic#jax.nn.softmax

def test_critic_net():
    bs = 64
    seq_len = 10
    obs_dim = 34
    head_len = jnp.array([16,2,16])
    net = CRITIC_Net(obs_dim, head_len)
    critic_params = net.init_params(256)
    obs=random.uniform(random.PRNGKey(0), shape=(bs,seq_len,obs_dim))
    value = net(critic_params, obs)
    print("value\n", value)

def test_policy_net():
    net=ACTOR_Net( 34, 12)
    policy_params = net.init_params(256)
    obs=random.uniform(random.PRNGKey(0), shape=(1024,34))
    act=net(policy_params, obs)
    print("act\n",act)

#test_critic_net()
#test_policy_net()
