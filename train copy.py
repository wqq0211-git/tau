import argparse
from core.ppo import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./trained/")          # Where to log diagnostics to
    parser.add_argument("--n_itr", type=int, default=10001, help="Number of iterations of the learning algorithm")
    parser.add_argument("--alr", type=float, default=0.001, help="actor Adam learning rate")
    parser.add_argument("--clr", type=float, default=0.001, help="critic Adam learning rate")#必须要比alr小，如果比actor还敏感会使优势与回报成反比（old_adv_batch =  old_ret_batch-old_val_batch）
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.96, help="MDP discount")
    parser.add_argument("--anneal", default=0.999, action='store_true', help="anneal rate for stddev")
    parser.add_argument("--std_dev", type=int, default=-2, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0, help="Coefficient for entropy regularization")
    parser.add_argument("--mirror_coeff", required=False, default=0, type=float, help="weight for mirror loss")
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", default=25,type=int, help="Batch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update")
    parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
    parser.add_argument("--max_traj_len", type=int, default=256, help="Max episode horizon")#每轮采样都是self.n_proc * self.max_traj_len个176
    parser.add_argument("--eval_freq", required=False, default=20, type=int, help="Frequency of performing evaluation")
    parser.add_argument(
        "--continued",
        nargs="?",
        const="auto",
        default="",
        type=str,
        help="Resume training. Use '--continued' to load from save_dir, or '--continued PATH' to load from another directory.",
    )
    parser.add_argument("--env_batch_size", required=False, default=1024, type=int, help="path to pretrained weights")
    parser.add_argument("--target_kl", required=False, default=0.2, type=float, help="limit on the kl div")
    parser.add_argument("--policy_expansion_n", required=False, default=256, type=int)
    parser.add_argument("--critic_expansion_n", required=False, default=256, type=int)
    args = parser.parse_args()
    algo = PPO(args=vars(args))
    algo.train()
