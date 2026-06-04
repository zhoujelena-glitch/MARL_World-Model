# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import importlib
import time
import warnings
import json
import argparse
import numpy as np
import gym
import torch

# === 你自己的 Key，按需保留/替换 ===
os.environ["WANDB_API_KEY"] = '8fb3b7c0c9edf8b119dcf508961f95dbccc06f2e'
os.environ["WANDB_TEAM"] = ""

warnings.filterwarnings('ignore')

from algorithms.utils import Config, LogClient, LogServer, mem_report
from algorithms.algo.main import OnPolicyRunner

# -------------- helpers: 将 dict/None 转为“可点属性”的对象，并做字段兜底 --------------
from types import SimpleNamespace
import ast, re

def _as_cfg_like(x):
    """把 dict/None 转成带 __dict__ 的对象（Config 优先），保证能点属性."""
    if isinstance(x, Config):
        return x
    if isinstance(x, SimpleNamespace):
        c = Config()
        c.__dict__.update(vars(x))
        return c
    if isinstance(x, dict):
        c = Config()
        c.__dict__.update(x)
        return c
    if x is None:
        return Config()
    return x  # 已经是可点属性的对象

def _parse_para_str(s: str):
    """更鲁棒的 --para 解析（支持 JSON / 单引号JSON / Python字面量 / k=v,k2=v2）"""
    if s is None:
        return {}
    s = s.strip()
    if not s:
        return {}
    # 1) 标准 JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) 单引号 JSON
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass
    # 3) Python 字面量
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 4) k=v,k2=v2
    d = {}
    for part in re.split(r'[,\n]+', s):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ('true', 'false'):
                v = (v.lower() == 'true')
            else:
                try:
                    v = json.loads(v)
                except Exception:
                    try:
                        v = ast.literal_eval(v)
                    except Exception:
                        pass
            d[k] = v
    return d

def _ensure_mlp_sizes(arg_cfg, in_dim: int, out_dim: int, default_hidden=(256, 256)):
    """把 MLP 的 sizes 强制对齐到 (in_dim, ..., out_dim)。"""
    sz = list(getattr(arg_cfg, "sizes", []))
    if not sz:
        sz = [in_dim, *default_hidden, out_dim]
    else:
        if sz[0] == -1 or sz[0] != in_dim:
            sz[0] = in_dim
        if sz[-1] != out_dim:
            sz[-1] = out_dim
    arg_cfg.sizes = list(map(int, sz))

def _ensure_agent_ppo_defaults(agent_args: Config):
    """DPPO/DMPO 共享的 actor/critic 训练缺省超参（仅在缺失时填充）"""
    defaults = dict(
        gamma=0.99, lamda=0.95, clip=0.20, target_kl=0.01,
        lr=3e-4, lr_v=1e-3,
        n_update_pi=80, n_update_v=80, n_minibatch=4, batch_size=4096,
        v_coeff=0.5, v_thres=1e9, entropy_coeff=0.0,
        advantage_norm=True, use_reduced_v=False, use_rtg=False,
        use_gae_returns=True, max_grad_norm=0.5, K_epochs=10,
    )
    for k, v in defaults.items():
        if not hasattr(agent_args, k):
            setattr(agent_args, k, v)

def _ensure_p_args_gnn_defaults(p_args: Config):
    """
    GraphConvolutionalModel 所需字段：
    n_conv, n_embedding, residual, edge_embed_dim, node_embed_dim,
    edge_hidden_size, node_hidden_size, reward_coeff, dropout
    """
    if hasattr(p_args, 'n_conv') and p_args.n_conv is None:
        delattr(p_args, 'n_conv')
    if not hasattr(p_args, 'n_conv'):
        p_args.n_conv = 2
    if not hasattr(p_args, 'n_embedding'):
        p_args.n_embedding = 128
    if not hasattr(p_args, 'residual'):
        p_args.residual = True
    if not hasattr(p_args, 'edge_embed_dim'):
        p_args.edge_embed_dim = 12
    if not hasattr(p_args, 'node_embed_dim'):
        p_args.node_embed_dim = 16
    if not hasattr(p_args, 'edge_hidden_size'):
        p_args.edge_hidden_size = [128, 128]
    if not hasattr(p_args, 'node_hidden_size'):
        p_args.node_hidden_size = [128, 128]
    if not hasattr(p_args, 'reward_coeff'):
        p_args.reward_coeff = 1.0
    if not hasattr(p_args, 'dropout'):
        p_args.dropout = 0.0
    return p_args

def _ensure_dmpo_runner_defaults(alg_args: Config):
    """Runner 在 model_based==True 时会读取的一系列字段"""
    if not hasattr(alg_args, 'n_traj'):                alg_args.n_traj = 4
    if not hasattr(alg_args, 'model_based'):           alg_args.model_based = True
    if not hasattr(alg_args, 'model_traj_length'):     alg_args.model_traj_length = 100
    if not hasattr(alg_args, 'model_error_thres'):     alg_args.model_error_thres = 0.10
    if not hasattr(alg_args, 'model_buffer_size'):     alg_args.model_buffer_size = 2000
    if not hasattr(alg_args, 'model_batch_size'):      alg_args.model_batch_size = 128
    if not hasattr(alg_args, 'n_model_update'):        alg_args.n_model_update = 5
    if not hasattr(alg_args, 'n_model_update_warmup'): alg_args.n_model_update_warmup = 10
    if not hasattr(alg_args, 'model_update_length'):   alg_args.model_update_length = 1
    if not hasattr(alg_args, 'model_validate_interval'): alg_args.model_validate_interval = 100
    if not hasattr(alg_args, 'model_length_schedule'): alg_args.model_length_schedule = [1, 1, 2, 2, 3]
    if not hasattr(alg_args, 'model_prob'):            alg_args.model_prob = 0.0

# ------------------------------- Args -------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='suez',
                        help="environment (add 'suez' for Unity canal)")
    parser.add_argument('--algo', type=str, default='DPPO',
                        help="algorithm (DMPO/IC3Net/CPPO/DPPO/IA2C)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument('--name', type=str, default='', help="additional name for logger")
    parser.add_argument('--para', type=str, default='{}', help="hyperparameter json string")
    parser.add_argument('--epoch', type=int, default=None,
                        help="number of training iterations (maps to alg_args.n_iter)")
    parser.add_argument('--unity_exe', type=str, default=None,
                        help="path to Unity player exe for canal scene")

    # === 与统一 wrapper 对齐的可选参数 ===
    parser.add_argument('--decision_interval', type=int, default=1,
                        help='act once every K steps; reuse previous action otherwise')
    parser.add_argument('--action_api', type=str, default='ppo9', choices=['ppo9', 'dict'],
                        help='use ppo9 (MultiDiscrete 9 per agent) or legacy dict')
    parser.add_argument('--relaunch_on_reset', action='store_true',
                        help='relaunch Unity on every env.reset()')

    args = parser.parse_args()
    # 强韧解析 --para
    args.para = _parse_para_str(args.para)

    # CUDA 不可用时自动降级
    if isinstance(args.device, str) and args.device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> fall back to CPU")
        args.device = 'cpu'

    return args

# -------------------------- Env factories --------------------------

def initEnv(input_args):
    """
    统一到合并后的 wrapper：
    Envs.suez_env_wrapper.SuezCanalGymWrapper(action_api='ppo9', decision_interval, relaunch_on_reset, unity_exe)
    """
    if input_args.env == 'suez':
        from Envs.suez_env_wrapper import SuezCanalGymWrapper
        def _make():
            return SuezCanalGymWrapper(
                unity_exe=input_args.unity_exe,
                relaunch_on_reset=bool(input_args.relaunch_on_reset),
                action_api=input_args.action_api,
                decision_interval=int(input_args.decision_interval),
                step_delay_sec=float(input_args.para.get('step_delay_sec', 0.0)),  # <== 新增
            )
        env_fn_train = _make
        env_fn_test  = _make
    else:
        env_fn_train = None
        env_fn_test  = None
    return env_fn_train, env_fn_test

# ---------------------------- Run/Env args ----------------------------

def getEnvArgs(input_args):
    env_args = Config()

    # —— Runner/Logger 依赖的基本标识（★必须包含 algo/env_name）——
    env_args.env = input_args.env
    env_args.env_name = input_args.env
    env_args.algo = input_args.algo
    env_args.algo_name = input_args.algo
    env_args.name = input_args.name

    # —— 资源/并发 ——
    env_args.n_env = 1
    env_args.n_cpu = 1
    env_args.n_gpu = 0

    # —— 透传给环境工厂/参考 ——
    env_args.device = input_args.device
    env_args.unity_exe = input_args.unity_exe
    env_args.action_api = getattr(input_args, 'action_api', 'ppo9')
    env_args.decision_interval = int(getattr(input_args, 'decision_interval', 1))
    env_args.relaunch_on_reset = bool(getattr(input_args, 'relaunch_on_reset', False))
    return env_args

def getRunArgs(input_args):
    run_args = Config()
    run_args.n_thread = 1
    run_args.parallel = False
    run_args.device = input_args.device

    run_args.n_cpu = 0.25
    run_args.n_gpu = 0
    run_args.debug = False
    run_args.test = False
    run_args.profiling = False
    run_args.name = f'standard{input_args.name}'

    run_args.radius_v = 1
    run_args.radius_pi = 1
    run_args.radius_p = 1

    run_args.init_checkpoint = None
    run_args.start_step = 0
    run_args.save_period = 1800
    run_args.log_period = int(20)
    run_args.seed = None
    return run_args

def initArgs(run_args, env_train, env_test, input_arg):
    import numpy as _np
    from gym.spaces import Discrete

    ref_env = env_train

    # ==== 选择配置模块 ====
    if input_arg.env == 'suez' and input_arg.algo == 'DPPO':
        config = importlib.import_module("algorithms.config.Suez_DPPO")
    elif input_arg.env == 'suez' and input_arg.algo == 'DMPO':
        # ⭐ 关键：DMPO 用专属配置
        config = importlib.import_module("algorithms.config.Suez_DMPO")
    else:
        if input_arg.env in ['eight', 'ring', 'catchup', 'slowdown', 'Grid',
                             'Monaco', 'PowerGrid', 'Real_Power', 'Pandemic', 'Large_city'] \
           or input_arg.algo in ['CPPO', 'DMPO', 'IC3Net', 'IA2C']:
            env_str = input_arg.env[0].upper() + input_arg.env[1:]
            config = importlib.import_module(f"algorithms.config.{env_str}_{input_arg.algo}")
        else:
            raise RuntimeError(f"Unsupported env/algo combo: env={input_arg.env}, algo={input_arg.algo}")

    # ==== 半径参数 ====
    if input_arg.env in ['catchup', 'slowdown']:
        run_args.radius_v = 4
        run_args.radius_pi = 1
        run_args.radius_p = 1
    if input_arg.env in ['Monaco', 'Grid', 'PowerGrid', 'Real_Power', 'Pandemic', 'Large_city']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1
    if input_arg.algo in ['CPPO']:
        run_args.radius_v = getattr(ref_env, "n_agents", 1)
        run_args.radius_pi = 1
        run_args.radius_p = 1

    # ==== 取配置 ====
    alg_args = config.getArgs(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)

    # ==== 推断真实维度 ====
    try:
        n_agents = int(getattr(ref_env, "n_agents", 0) or ref_env.observation_space.shape[0])
        obs_dim = int(getattr(ref_env, "obs_size", getattr(ref_env, "n_s", None)) or ref_env.observation_space.shape[-1])
        # 每站动作维（若 env 是 MultiDiscrete，就取最大分支）
        if hasattr(ref_env.action_space, "n"):   # Discrete
            act_n = int(ref_env.action_space.n)
        elif hasattr(ref_env.action_space, "nvec"):  # MultiDiscrete
            act_n = int(np.max(ref_env.action_space.nvec))
        else:
            act_n = 9
    except Exception:
        n_agents, obs_dim, act_n = 10, 256, 9

    # ==== 确保存在 agent_args / pi_args / v_args ====
    if not hasattr(alg_args, "agent_args") or alg_args.agent_args is None:
        alg_args.agent_args = Config()
    if not hasattr(alg_args, "pi_args") or alg_args.pi_args is None:
        alg_args.pi_args = Config()
    if not hasattr(alg_args, "v_args") or alg_args.v_args is None:
        alg_args.v_args = Config()

    # ==== 强制对齐 MLP 输入输出 ====
    _ensure_mlp_sizes(alg_args.pi_args, in_dim=obs_dim, out_dim=act_n)
    _ensure_mlp_sizes(alg_args.v_args,  in_dim=obs_dim, out_dim=1)

    # ==== 写回 agent_args（关键：强制 Discrete(action_n)）====
    from gym.spaces import Discrete
    agent_args = alg_args.agent_args
    agent_args.n_agent          = n_agents
    agent_args.observation_dim  = obs_dim
    agent_args.n_action         = act_n
    agent_args.action_dim       = act_n
    agent_args.action_space     = Discrete(act_n)
    agent_args.pi_args          = alg_args.pi_args
    agent_args.v_args           = alg_args.v_args
    agent_args.p_args           = _as_cfg_like(getattr(alg_args, "p_args", getattr(agent_args, "p_args", None)))

    # 缺省字段兜底
    defaults = dict(
        squeeze=False,
        discrete=True,
        share_param=True if n_agents > 1 else False,
    )
    for k, v in defaults.items():
        if not hasattr(agent_args, k):
            setattr(agent_args, k, v)

    # 顶层也挂成 Discrete
    alg_args.n_agent         = n_agents
    alg_args.observation_dim = obs_dim
    alg_args.action_space    = Discrete(act_n)
    alg_args.discrete        = True

    # 邻接矩阵（失败则单位阵）
    try:
        sids = list(getattr(ref_env, "station_ids", range(n_agents)))
        idx = {sid: i for i, sid in enumerate(sids)}
        obs = ref_env.core.get_observations() if hasattr(ref_env, "core") else {}
        adj = np.eye(n_agents, dtype=np.float32)
        for sid in sids:
            for nb in (obs.get(sid, {}) or {}).get("neighbors", []) or []:
                if nb in idx:
                    i, j = idx[sid], idx[nb]
                    adj[i, j] = adj[j, i] = 1.0
        agent_args.adj = adj
    except Exception:
        agent_args.adj = _np.eye(n_agents, dtype=_np.float32)

    # ==== 兜底循环参数 ====
    # OnPolicyRunner 读 max_episode_len（不是 max_ep_len）
    if not hasattr(alg_args, "max_episode_len") or alg_args.max_episode_len is None:
        alg_args.max_episode_len = 400
    if not hasattr(alg_args, "rollout_length") or alg_args.rollout_length is None:
        alg_args.rollout_length = 200
    if not hasattr(alg_args, "n_iter") or alg_args.n_iter is None:
        alg_args.n_iter = 1000
    if not hasattr(alg_args, "n_inner_iter") or alg_args.n_inner_iter is None:
        alg_args.n_inner_iter = 10
    if not hasattr(alg_args, "test_length") or alg_args.test_length is None:
        alg_args.test_length = 1
    if not hasattr(alg_args, "test_interval") or alg_args.test_interval is None:
        alg_args.test_interval = 100

    # ==== agent 超参缺省 ====
    _ensure_agent_ppo_defaults(agent_args)

    # ==== DMPO 额外必需超参兜底 ====
    if input_arg.algo == 'DMPO':
        if not hasattr(agent_args, 'lr_p') or agent_args.lr_p is None:
            agent_args.lr_p = 1e-3
        agent_args.p_args = _as_cfg_like(agent_args.p_args)
        agent_args.p_args = _ensure_p_args_gnn_defaults(agent_args.p_args)
        _ensure_dmpo_runner_defaults(alg_args)

    return alg_args

def override(alg_args, run_args, env_fn_train, input_args, agent_fn):
    # 训练入口
    alg_args.env_fn = env_fn_train

    # 这里假定 initArgs 已确保 agent_args 存在
    agent_args = alg_args.agent_args

    if run_args.debug:
        alg_args.model_batch_size = 4
        alg_args.max_episode_len = 5
        alg_args.rollout_length = 5
        alg_args.test_length = 1
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 10

    if run_args.test:
        alg_args.n_warmup = 0
        alg_args.n_test = 10

    if run_args.profiling:
        alg_args.model_batch_size = 128
        alg_args.n_warmup = 0
        if getattr(alg_args, "agent_args", None) is None or getattr(alg_args.agent_args, "p_args", None) is None:
            alg_args.n_iter = 10
        else:
            alg_args.n_iter = 10
            alg_args.model_buffer_size = 1000
            alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.max_episode_len = 400
        alg_args.rollout_length = 400
        alg_args.test_length = 1
        alg_args.test_interval = 100

    if run_args.seed is None:
        run_args.seed = int(time.time() * 1000) % 65536

    agent_args.parallel = run_args.parallel
    agent_args.lable_name = input_args.algo + input_args.name

    # 覆盖命令行传入的超参（支持 agent_args.p_args.n_conv 这类嵌套）
    for key, val in input_args.para.items():
        key_ls = key.split('.')
        *pre_key_ls, key_last = key_ls
        target_args = alg_args
        for pre_key in pre_key_ls:
            cur = target_args.__dict__.get(pre_key, None)
            cur = _as_cfg_like(cur)
            target_args.__dict__[pre_key] = cur
            target_args = cur
        target_args.__dict__[key_last] = val

    # 再次兜底：防止 para 把 p_args 写成 dict，统一回“可点属性”，并补齐必需字段
    alg_args.agent_args.p_args = _as_cfg_like(getattr(alg_args.agent_args, 'p_args', None))
    _ensure_agent_ppo_defaults(alg_args.agent_args)
    if input_args.algo == 'DMPO':
        _ensure_p_args_gnn_defaults(alg_args.agent_args.p_args)
        _ensure_dmpo_runner_defaults(alg_args)

    # 运行名
    run_args.name = '{}_{}'.format(agent_fn.__name__, run_args.seed)
    return alg_args, run_args

# ------------------------------- Main -------------------------------

class DMPOProxy:
    """
    轻量代理：在 Runner 调 self.agent.updateAgent(trajs) 之前，
    自动调用一次世界模型 self.agent.updateModel(trajs, length=model_len)。
    其他属性/方法均透传到原始 agent。
    """
    def __init__(self, base_agent, alg_args):
        self._a = base_agent
        self._alg = alg_args

    def __getattr__(self, name):
        return getattr(self._a, name)

    def updateAgent(self, trajs):
        length = int(getattr(self._alg, 'model_len', getattr(self._alg, 'model_traj_length', 1)))
        try:
            if hasattr(self._a, 'updateModel'):
                # 关键修复：先把轨迹裁成 length，再喂给世界模型
                short_trajs = [t.getFraction(length=length) for t in trajs]
                self._a.updateModel(short_trajs, length=length)
        except Exception as e:
            print(f"[DMPOProxy] updateModel failed: {e}")
        return self._a.updateAgent(trajs)


if __name__ == "__main__":
    input_args = parse_args()

    # 选择 agent
    if input_args.algo == 'IA2C':
        from algorithms.algo.agent.IA2C import IA2C as agent_fn
    elif input_args.algo == 'IC3Net':
        from algorithms.algo.agent.IC3Net import IC3Net as agent_fn
    elif input_args.algo == 'DPPO':
        from algorithms.algo.agent.DPPO import DPPOAgent as agent_fn
    elif input_args.algo == 'CPPO':
        from algorithms.algo.agent.CPPO import CPPOAgent as agent_fn
    elif input_args.algo == 'DMPO':
        from algorithms.algo.agent.DMPO import DMPOAgent as agent_fn
    else:
        raise RuntimeError(f"Unsupported algo: {input_args.algo}")

    # env_args 需要传 input_args
    env_args = getEnvArgs(input_args)

    # —— 兜底：强制补齐关键别名，防止遗漏导致 Runner 报错 ——
    for k, v in {
        "algo": input_args.algo,
        "algo_name": input_args.algo,
        "env": input_args.env,
        "env_name": input_args.env,
    }.items():
        setattr(env_args, k, v)

    env_fn_train, env_fn_test = initEnv(input_args)
    env_train = env_fn_train()
    env_test = env_train

    # 运行超参
    run_args = getRunArgs(input_args)

    # 算法超参（含网络结构）
    alg_args = initArgs(run_args, env_train, env_test, input_args)
    if getattr(input_args, 'epoch', None) is not None:
        alg_args.n_iter = int(input_args.epoch)

    # 应用命令行覆盖（含安全处理 p_args）
    alg_args, run_args = override(alg_args, run_args, env_fn_train, input_args, agent_fn)

    # 显式 GPU 可见性（按需）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    # 日志 & Agent
    logger = LogServer({'run_args': run_args, 'algo_args': alg_args},
                       mute=run_args.debug or run_args.test or run_args.profiling)
    logger = LogClient(logger)

    # === 关键：把 env_args 传给 Agent（DMPO 需要用到 env 相关信息） ===
    agent = agent_fn(logger, run_args.device, alg_args.agent_args, env_args)

    # 如果是 DMPO，用代理把 updateModel 串到 updateAgent 前执行
    if input_args.algo == 'DMPO':
        agent = DMPOProxy(agent, alg_args)
        print("[INFO] DMPO enabled: rollout -> updateModel -> updateAgent (via DMPOProxy).")

    print(f"n_threads {torch.get_num_threads()}")
    print(f"n_gpus {torch.cuda.device_count()}")

    # 统一：两个分支都用同一 OnPolicyRunner 调用签名（包含 env_args）
    runner = OnPolicyRunner(
        logger=logger,
        run_args=run_args,
        alg_args=alg_args,
        agent=agent,
        env_learn=env_train,
        env_test=env_test,
        env_args=env_args,           # ← 必传
    )

    if run_args.profiling:
        import cProfile
        cProfile.run(
            "runner.run()",
            filename=f'device{run_args.device}_parallel{run_args.parallel}.profile'
        )
    else:
        runner.run()
