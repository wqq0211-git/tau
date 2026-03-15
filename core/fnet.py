import jax
import jax.numpy as jnp
from jax import random
import time

class FR_EXP_Net():
    def __init__(self, obs_dim, act_dim, n):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n = n
        self.c=jnp.arange(1,self.n+1,1).reshape((1,1,1,n))
    def init_params(self):
        key = random.PRNGKey(int(time.time()))
        # 初始化策略网络参数
        return {
            'scale': random.uniform(key, shape=(1, self.act_dim, self.obs_dim), minval=0.0, maxval=1.0),
            'fourier_expansion_sin_coefficient': random.uniform(key, (1, self.act_dim, self.obs_dim, self.n)) - 0.5,
            'fourier_expansion_cos_coefficient': random.uniform(key, (1, self.act_dim, self.obs_dim, self.n)) - 0.5,
            'fourier_expansion_bias': random.uniform(key, (1, self.act_dim)) - 0.5
        }
    def __call__(self, policy_params, obs):
        reshaped_obs=jnp.expand_dims(obs, axis=1)
        code=jnp.expand_dims(jax.nn.relu(policy_params['scale'])*reshaped_obs, axis=3)*self.c
        sin_info = jnp.sum(jnp.sin(code)*policy_params['fourier_expansion_sin_coefficient'],axis=-1)
        cos_info = jnp.sum(jnp.cos(code)*policy_params['fourier_expansion_cos_coefficient'],axis=-1)
        action_mean = jnp.mean(sin_info, axis=-1) + jnp.mean(cos_info, axis=-1) + policy_params['fourier_expansion_bias']
        return action_mean

class CRITIC_Net():
    def __init__(self, obs_dim, n, head_len):
        self.obs_dim = obs_dim
        self.n = n
        self.c = jnp.arange(n)
        self.a = jnp.power(1.3, jnp.arange(1,n+1,1)).reshape((1,1,1,n))
        self.sqrt_dk = jnp.sqrt(head_len)
        self.head_len = head_len
    def init_params(self):
        key = random.PRNGKey(int(time.time()))
        # 初始化价值网络参数
        return {
            'query_weights': random.uniform(key, (1,1, self.obs_dim)),
            'key_weights' : random.uniform(key, (1,1, self.obs_dim)),
            'value_weights': random.uniform(key, (1,1, self.obs_dim)),
            'out_linear_weights': random.uniform(key, (1,1,self.obs_dim)),
            'query_bias':0.0, 'key_bias':0.0, 'value_bias':0.0,'out_linear_bias':0.0,
            'expansion_coefficient': random.uniform(key, (1, 1, self.obs_dim, self.n)),
            #'bias': jnp.expand_dims(random.uniform(key, self.obs_dim), axis=0)
        }
    def __call__(self, critic_params, obs):
        attention_obs = jnp.zeros_like(obs)
        query = critic_params['query_weights'] * obs + critic_params['query_bias']
        key = critic_params['key_weights'] * obs + critic_params['key_bias']
        value = critic_params['value_weights'] * obs + critic_params['value_bias']
        idx=0
        for i in range(self.head_len.shape[0]):
            query_e=query[:,:,idx:idx+self.head_len[i]]
            key_e=key[:,:,idx:idx+self.head_len[i]]
            value_e=value[:,:,idx:idx+self.head_len[i]]
            scores = jnp.matmul(query_e, jnp.swapaxes(key_e, -1, -2)) / self.sqrt_dk[i]
            attention_weights = jax.nn.softmax(scores, axis=-1)
            attention_obs=attention_obs.at[:,:,idx:idx+self.head_len[i]].set(jnp.matmul(attention_weights, value_e))
            idx += self.head_len[i]
        encoded_obs=critic_params['out_linear_weights'] * attention_obs + critic_params['out_linear_bias']
        assert encoded_obs.shape == obs.shape
        vandermonde_matrix=jnp.expand_dims(encoded_obs, axis=-1)**self.c
        critic = jnp.mean(vandermonde_matrix*self.a*critic_params['expansion_coefficient'],axis=(-2,-1))
        #print("critic_params['weights']", critic_params['weights'])
        #print("vandermonde_matrix", vandermonde_matrix)
        return critic#jax.nn.softmax
'''
def test_multi_head_attention():
    bs = 4
    seq_len = 10
    obs_dim = 34
    n=8
    head_len = jnp.array([18,16])
    net = CRITIC_Net(obs_dim, n, head_len)
    critic_params = net.init_params()
    #obs = jnp.array([jnp.arange(10), 10 + jnp.arange(10), 20 + jnp.arange(10), 30 + jnp.arange(10)])
    obs=jnp.ones((bs,seq_len,obs_dim))
    output = net(critic_params, obs)
    print("output\n", output)

test_multi_head_attention()
'''
'''
net=FR_EXP_Net( 3, 2, 4)
policy_params = net.init_params()
obs=random.uniform(random.PRNGKey(0), shape=(5,3))
net.forward(policy_params, obs)
#print(act.shape)
'''
'''
net=CRITIC_Net( 10, 3)
critic_params = net.init_params()
obs=jnp.array([jnp.arange(10),10+jnp.arange(10),20+jnp.arange(10),30+jnp.arange(10)])
v=net(critic_params, obs)
print(v.shape)
'''
'''
bs=5
a=jnp.arange(30).reshape((5,2,3))
b=jnp.tile(jnp.arange(3),(5,1,1))
print(b.shape)
print("a*b\n",(a*b).shape)
d=jnp.expand_dims(a*b, axis=3)
c=jnp.arange(4).reshape((1,1,1,4))
print("d.shape: ",d.shape)
print("shape ",(d*c).shape)
print("f",(d*c).sum(axis=(2,3)).shape)
'''