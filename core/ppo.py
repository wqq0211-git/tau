"""Proximal Policy Optimization (clip objective)."""
import os
import time
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, tree_util, value_and_grad
import pickle
from jax.scipy.stats import norm
from core.convnet import ACTOR_Net,CRITIC_Net
from core.batch_walk import WalkEnv

def BatchSampler(num_samples, minibatch_size):
    key = random.PRNGKey(0)
    random_indices = random.permutation(key, num_samples)
    num_batches = num_samples // minibatch_size
    batches = [random_indices[i * minibatch_size:(i + 1) * minibatch_size] for i in range(num_batches)]
    return batches

class DataSet:
    def __init__(self, batch_size, max_trace_len, obs_dim, act_dim):
        self.val = jnp.zeros((batch_size, max_trace_len))
        self.ret = jnp.zeros((batch_size, max_trace_len))
        self.obs = jnp.zeros((batch_size, max_trace_len, obs_dim))
        self.act = jnp.zeros((batch_size, max_trace_len, act_dim))
        self.old_act_mu = jnp.zeros((batch_size, max_trace_len, act_dim))
        self.mean_trace_val = 0  #只和state有关（状态-动作的期望）
        self.mean_trace_ret = 0  # 还和动作有关，从各条轨迹中提取（state,action）对应的价值
        self.trace_len = 0
        self.batch_size = batch_size
        self.mean_val = 0
        self.mean_ret = 0


class PPO:
    def __init__(self, args):
        self.gamma = args['gamma']
        self.lam = args['lam']
        self.alr = args['alr']
        self.clr = args['clr']
        self.eps = args['eps']
        self.ent_coeff = args['entropy_coeff']
        self.mirror_coeff = args['mirror_coeff']
        self.clip = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.env_batch_size = args['env_batch_size']
        self.epochs = args['epochs']
        self.max_traj_len = args['max_traj_len']
        self.use_gae = args['use_gae']
        self.grad_clip = args['max_grad_norm']
        self.eval_freq = args['eval_freq']
        self.save_path = args['save_dir']
        self.n_itr = args['n_itr']
        self.anneal_rate = args['anneal']
        self.target_kl = args['target_kl']
        self.std = jnp.exp(args['std_dev'])
        self.entropy_penalty = -jnp.mean(-0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.exp(args['std_dev']) ** 2))
        self.critic_expansion_n = args['critic_expansion_n']
        self.policy_expansion_n = args['policy_expansion_n']
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        self.approx_kl_div = 0
        #self.coefficient = jnp.array([100,0.1e-6,0.4])
        self.coefficient = jnp.array([0.8, 0.2])
        self.eval_fn = os.path.join(self.save_path, 'eval.txt')
        self.env = WalkEnv(self.env_batch_size)
        self.policy = ACTOR_Net(self.env.obs_dim,self.env.act_dim,self.policy_expansion_n)  # self.env.obs_dim, self.env.act_dim, fixed_std=np.exp(args.std_dev), bounded=False)
        #FR_EXP_Net(self.env.obs_dim, self.env.act_dim,self.policy_expansion_n)
        self.critic = CRITIC_Net(self.env.obs_dim,self.critic_expansion_n)  # 估算当前状态的价值（即reward）

        resume_dir = self._resolve_resume_dir(args.get('continued', ''))
        if resume_dir is not None:
            policy_path = os.path.join(resume_dir, "best_policy.pkl")
            critic_path = os.path.join(resume_dir, "best_critic.pkl")
            if os.path.exists(policy_path) and os.path.exists(critic_path):
                print(f"load models from {resume_dir}...")
                with open(policy_path, 'rb') as f:
                    self.policy_params = pickle.load(f)
                with open(critic_path, 'rb') as f:
                    self.critic_params = pickle.load(f)
            else:
                print(f"resume requested, but weights were not found in {resume_dir}. start from scratch.")
                self.policy_params = self.policy.init_params()
                self.critic_params = self.critic.init_params()
        else:
            self.policy_params = self.policy.init_params()  # self.policy_expansion_n
            self.critic_params = self.critic.init_params()
        self.num_samples = self.max_traj_len * self.env_batch_size

    def _resolve_resume_dir(self, continued):
        if isinstance(continued, str):
            continued = continued.strip()
            if continued.lower() in ("", "0", "false", "no", "none"):
                return None
            if continued.lower() in ("1", "true", "yes", "auto"):
                return self.save_path
            return continued
        if continued:
            return self.save_path
        return None

    def _save_checkpoint(self, prefix):
        with open(os.path.join(self.save_path, f"{prefix}_policy.pkl"), 'wb') as f:
            pickle.dump(self.policy_params, f)
        with open(os.path.join(self.save_path, f"{prefix}_critic.pkl"), 'wb') as f:
            pickle.dump(self.critic_params, f)


    def upper_toeplitz_geo(self, n):
        i = jnp.arange(n)[:, None]
        j = jnp.arange(n)[None, :]
        # 超对角线阶数
        k = j - i
        return jnp.where(k >= 0, self.gamma ** k, 0.0)

    def sample(self, add_noise=True):
        obs = self.env.reset()
        pool = DataSet(self.env_batch_size, self.max_traj_len, self.env.obs_dim, self.env.act_dim)
        dones = jnp.zeros((self.env_batch_size,1), dtype=bool)
        while pool.trace_len<self.max_traj_len and not dones.any():

            prestates = obs.copy()
            actions = self.policy(self.policy_params, prestates)  #输出batch_size*12二维数组

            pool.old_act_mu = pool.old_act_mu.at[:, pool.trace_len].set(actions)
            if add_noise:
                samples = jax.random.normal(
                    jax.random.PRNGKey(int(time.time())),
                    shape=(self.env.batch_size, self.env.act_dim),
                )
                actions += self.std * samples #self.act = jnp.zeros((batch_size, self.act_dim))

            values = self.critic(self.critic_params, prestates)
            
            obs, rewards, dones = self.env.step(actions)
            pool.obs = pool.obs.at[:, pool.trace_len].set(prestates)
            pool.act = pool.act.at[:, pool.trace_len].set(actions)
            pool.ret = pool.ret.at[:, pool.trace_len].set(rewards)
            pool.val = pool.val.at[:, pool.trace_len].set(values)
            #self.pool.ret = self.pool.ret.at[:, self.pool.trace_len].set((rewards * (1 - self.gamma ** (self.pool.trace_len+1)) / (1 - self.gamma)))
            pool.trace_len+=1
        #print("obs.shape", pool.obs.shape)
        pool.ret = pool.ret.at[:, pool.trace_len].set(jnp.where(dones, -5, pool.ret[:, pool.trace_len]))#包含了提前终止惩罚
        mat = self.upper_toeplitz_geo(pool.trace_len)[None, :, :]
        pool.ret = pool.ret.at[:, :pool.trace_len].set(
            jnp.squeeze(mat @ jnp.expand_dims(pool.ret[:, :pool.trace_len], -1)))
        pool.mean_ret = jnp.mean(pool.ret[:, :pool.trace_len])
        pool.mean_val = jnp.mean(pool.val[:, :pool.trace_len])
        return pool
        #jax.debug.print("self.pool.values: {}", self.pool.values)


    def calcu_adv(self, policy_params, critic_params,  old_obs_batch,old_act_batch,old_ret_batch,old_val_batch,old_act_mu):
        
        reshaped_old_obs_batch = jnp.reshape(old_obs_batch, (-1, self.env.obs_dim))
        #reshaped_mirror_old_obs_batch=jnp.reshape(mirror_old_obs_batch, (-1, self.env.obs_dim))

        old_adv_batch =  old_ret_batch-old_val_batch
        old_adv_batch = (old_adv_batch - old_adv_batch.mean()) / (old_adv_batch.std() + self.eps)#否则ret会踩着val,val越来越低，二者差距越来越大
        '''
        trace_adv = jnp.sum(old_adv_batch, axis=1)
        mean_trace_adv = jnp.mean(trace_adv)
        jax.debug.print("trace_adv: {}", mean_trace_adv)
        '''
        new_val = self.critic(critic_params,reshaped_old_obs_batch)
        new_val = jnp.reshape(new_val, old_ret_batch.shape)

        critic_mse_loss = jnp.mean(jnp.square(old_ret_batch - new_val))
        critic_adv = -critic_mse_loss
        
        new_act_mu = self.policy(policy_params, reshaped_old_obs_batch)#.reshape((old_act_mu.shape[0], old_act_mu.shape[1], self.env.act_dim))
        #mirror_mse_loss=jnp.mean(jnp.square(new_act_mu - mirror_new_act_mu))
        #mirror_adv = -mirror_mse_loss

        new_act_mu = jnp.reshape(new_act_mu, old_act_batch.shape)


        log_probs = jnp.mean(norm.logpdf(old_act_batch, loc=new_act_mu, scale=self.std), axis=-1)
        old_log_probs = jnp.mean(norm.logpdf(old_act_batch, loc=old_act_mu, scale=self.std), axis=-1)  #policy网络作出action_batch决策的对数概率
        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)  

        clip_ratio = jnp.clip(ratio, 1.0 - self.clip, 1.0 + self.clip)
        actor_adv = jnp.minimum(ratio* old_adv_batch, clip_ratio* old_adv_batch)#old_adv_batch>0时ratio和clip_ratio取最小，old_adv_batch<0时ratio和clip_ratio取最大
        #actor_adv = jnp.sum(actor_adv,axis=-1)
        actor_adv = jnp.mean(actor_adv)#-self.pool.mean_trace_ret
        #jax.debug.print("actor_adv:\n {}", actor_adv*self.coefficient[0])
        #jax.debug.print("critic_adv:\n {}", critic_adv*self.coefficient[1])
        #jax.debug.print("mirror_adv:\n {}", mirror_adv*self.coefficient[2])
        #只有actor_adv可能是正数
        return jnp.dot(jnp.array([actor_adv,critic_adv]),self.coefficient)#actor_adv + critic_adv*self.critic_coefficient + mirror_adv*self.mirror_coefficient

    def train(self):
        curr_anneal = 1.0
        highest_reward = 0
        mean_ret = []
        mean_adv=[]
        trace_len = []
        itr = -1
        try:
            for itr in range(self.n_itr):  #self.n_itr
                if highest_reward > 0.9 and curr_anneal > 0.5:
                    curr_anneal *= self.anneal_rate
                pool = self.sample(add_noise=True)

                total_adv = 0
                #continue_training = True
                adv_n = 0
                for epoch in range(self.epochs):
                    sampler = BatchSampler(self.env_batch_size, self.minibatch_size)
                    for indices in sampler:

                        old_obs_batch = pool.obs[indices,:pool.trace_len]
                        old_act_batch = pool.act[indices,:pool.trace_len]
                        old_ret_batch = pool.ret[indices,:pool.trace_len]  # jnp.floor(indices / self.max_traj_len).astype(int)
                        old_val_batch = pool.val[indices,:pool.trace_len]
                        old_act_mu = pool.old_act_mu[indices,:pool.trace_len]
                        adv, grads = (value_and_grad(self.calcu_adv, argnums=(0,1))
                                      (self.policy_params, self.critic_params, old_obs_batch,old_act_batch,old_ret_batch,old_val_batch,old_act_mu,))

                        self.policy_params = tree_util.tree_map(lambda p, g: p + self.alr * g, self.policy_params, grads[0])
                        self.critic_params = tree_util.tree_map(lambda p, g: p + self.clr * g, self.critic_params, grads[1])
                        '''
                        for keys in self.policy_params.keys():
                            jax.debug.print("policy_params {}:{}", keys, jnp.mean(self.policy_params[keys]/grads[0][keys]))
                        for keys in self.critic_params.keys():
                            jax.debug.print("critic_params {}:{}", keys, jnp.mean(self.critic_params[keys]/grads[1][keys]))
                        '''

                        adv_n += 1
                        total_adv += adv

                if itr % 10 == 0:
                    print("********** Iteration {} ************".format(itr))
                    print("trace_lens: ", pool.trace_len)
                    print("mean_ret: ", pool.mean_ret)
                    print("mean_val: ", pool.mean_val)
                    print("mean_adv: ", total_adv / adv_n)
                    print()

                if itr % self.eval_freq == 0:
                    pool = self.sample(add_noise=False)
                    # Eval diagnostics do not need the full env batch; a small slice avoids OOM.
                    eval_batch_size = min(self.minibatch_size, pool.obs.shape[0])
                    adv = self.calcu_adv(
                        self.policy_params,
                        self.critic_params,
                        pool.obs[:eval_batch_size],
                        pool.act[:eval_batch_size],
                        pool.ret[:eval_batch_size],
                        pool.val[:eval_batch_size],
                        pool.old_act_mu[:eval_batch_size],
                    )
                    trace_len.append(pool.trace_len)
                    mean_ret.append((pool.mean_ret-10)*10)
                    mean_adv.append(adv)
                    plt.clf()
                    xlabel = [i * self.eval_freq for i in range(len(trace_len))]
                    #plt.plot(xlabel, trace_len, color='blue', marker='o', label='trace_lens')
                    #plt.plot(xlabel, mean_values, color='blue', marker='o', label='mean_values')
                    plt.plot(xlabel, mean_ret, color='green', marker='o', label='mean_ret')
                    #plt.plot(xlabel, mean_adv, color='red', marker='o', label='mean_adv')
                    plt.legend()
                    plt.savefig(os.path.join(self.save_path, "eval.jpg"), bbox_inches='tight')
                    if highest_reward < pool.mean_ret:
                        highest_reward = pool.mean_ret
                        self._save_checkpoint("best")
                        print("save best model:", float(pool.mean_ret))
                        print()
                    else:
                        self._save_checkpoint("last")
                        print("save last model:", float(pool.mean_ret))
                        print()
        except KeyboardInterrupt:
            self._save_checkpoint("interrupt")
            print("\nKeyboardInterrupt at iteration {}. Saved interrupt_policy.pkl and interrupt_critic.pkl.".format(itr))
            raise
