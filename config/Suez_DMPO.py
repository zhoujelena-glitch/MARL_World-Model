# -*- coding: utf-8 -*-
from algorithms.utils import Config
import numpy as np
from gym.spaces import Discrete
import torch
import torch.nn as nn
from algorithms.models import MLP
import os


def _infer_sizes(env):
    n_agent = int(getattr(env, "n_agents", 10))
    obs_dim = getattr(env, "obs_size", getattr(env, "n_s", None))
    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[-1])
    # 动作维：优先 Discrete.n；MultiDiscrete 取最大分支
    act_n = 9
    aspace = getattr(env, "action_space", None)
    if hasattr(aspace, "n"):
        act_n = int(aspace.n)
    elif hasattr(aspace, "nvec"):
        act_n = int(np.max(aspace.nvec))
    return int(n_agent), int(obs_dim), int(act_n)


def _adj_from_env(env, n_agent):
    import numpy as np
    adj = np.eye(n_agent, dtype=np.float32)
    try:
        sids = list(getattr(env, "station_ids", range(n_agent)))
        idx = {sid: i for i, sid in enumerate(sids)}
        stations = getattr(getattr(env, "core", env), "stations", {})
        for sid in sids:
            st = stations.get(sid)
            if not st:
                continue
            for nb in getattr(st, "neighbor_ids", []):
                if nb in idx:
                    i, j = idx[sid], idx[nb]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    except Exception:
        pass
    return adj


def _dreamer_vm_defaults(scale: str = "standard"):
    """
    参考 DreamerV3 默认，给出三档配置以兼顾显存：
      - tiny:     轻量快速调试
      - standard: 建议默认（接近论文中常用配置）
      - large:    更接近官方大配置（显存占用高）
    """
    scale = str(scale).lower().strip()
    if scale == "tiny":
        return dict(
            deter=1024, hidden=512, blocks=4,
            stoch_groups=16, stoch_classes=32, enc_img_dim=256
        )
    if scale == "large":
        return dict(
            deter=4096, hidden=2048, blocks=8,
            stoch_groups=32, stoch_classes=32, enc_img_dim=512
        )
    # standard
    return dict(
        deter=2048, hidden=1024, blocks=8,
        stoch_groups=32, stoch_classes=32, enc_img_dim=512
    )


def getArgs(radius_p, radius_v, radius_pi, env):
    # === 从环境推断规模 ===
    n_agent, obs_dim, act_n = _infer_sizes(env)

    # === 顶层算法配置（Runner/Logger 常读） ===
    alg = Config()
    alg.n_agent         = n_agent
    alg.observation_dim = obs_dim
    alg.action_space    = Discrete(act_n)
    alg.discrete        = True

    # ===== 训练循环参数 =====
    # 这些保持与你现有流程一致；你可以按需微调
    alg.n_iter               = 25000
    alg.n_inner_iter         = 20
    alg.rollout_length       = 180
    alg.test_length          = 180
    alg.max_episode_len      = 180
    alg.n_warmup             = 10
    alg.n_test               = 1
    alg.test_interval        = 10

    # ===== 模型学习（DMPO）必需字段 =====
    alg.model_based             = True
    alg.n_traj                  = 512
    # DreamerV3 使用较长链，建议 64；你原先 15 也能跑，但重构更稳的是 64
    alg.model_traj_length       = 64
    alg.model_error_thres       = 1e-5
    alg.model_buffer_size       = 20
    # DreamerV3 常用较小 batch 的序列（16 或 32）；显存足够可以 32
    alg.model_batch_size        = 16
    # 每次外循环中世界模型的更新步数（可视算力而定）
    alg.n_model_update          = int(1e3)
    alg.n_model_update_warmup   = int(2e3)
    # 单次传给 world model 的截断序列长度（Dreamer 用 64）
    alg.model_update_length     = 64
    alg.model_validate_interval = 10
    alg.model_length_schedule   = None
    # 按你的管线：混合真实与模型数据；此处先保留 0.5
    alg.model_prob              = 0.5

    # 【可选：预训练世界模型开关与路径（默认关）】
    alg.load_pretrained_model = False
    alg.pretrained_model = None

    # ===== PPO/DPPO 超参（拷给 agent_args）=====
    HP = Config()
    HP.gamma            = 0.99
    HP.lamda            = 0.95
    HP.clip             = 0.20
    HP.target_kl        = 0.02

    HP.lr               = 5e-4     # actor lr
    HP.lr_v             = 3e-4     # critic lr

    HP.n_update_pi      = 8
    HP.n_update_v       = 10
    HP.n_minibatch      = 1

    HP.v_coeff          = 1.0
    HP.v_thres          = 0.
    HP.entropy_coeff    = 0.05
    HP.advantage_norm   = True
    HP.use_reduced_v    = True
    HP.use_rtg          = True
    HP.use_gae_returns  = False

    # 图半径（按 Runner 约定）
    HP.radius_v         = int(radius_v)
    HP.radius_pi        = int(radius_pi)

    # ===== 网络结构（策略/价值）=====
    pi_args = Config()
    pi_args.network     = MLP
    pi_args.activation  = nn.ReLU
    pi_args.sizes       = [-1, 128, 128, act_n]
    pi_args.squash      = False

    v_args = Config()
    v_args.network      = MLP
    v_args.activation   = nn.ReLU
    v_args.sizes        = [-1, 128, 128, 1]

    # ===== agent_args：DMPO 需要世界模型参数 =====
    agent_args = Config()
    agent_args.n_agent          = n_agent
    agent_args.observation_dim  = obs_dim
    agent_args.action_space     = Discrete(act_n)
    agent_args.discrete         = True
    agent_args.share_param      = True if n_agent > 1 else False
    agent_args.action_dim       = act_n
    agent_args.n_action         = act_n
    agent_args.squeeze          = False

    # PPO 超参复制进 agent_args
    for k, v in HP.__dict__.items():
        setattr(agent_args, k, v)

    # 世界模型优化器 lr（DreamerV3 常用 3e-4）
    agent_args.lr_p = 3e-4

    # ===== 世界模型结构参数（DreamerV3 原版风格） =====
    p_args = Config()
    # —— 档位（可用环境变量 DREAMER_SCALE 控制：tiny/standard/large）——
    scale = os.environ.get("DREAMER_SCALE", "standard")
    defaults = _dreamer_vm_defaults(scale)

    p_args.model_type     = 'dreamerv3'       # 切到 DreamerV3WorldModel

    # RSSM/MLP 规模
    p_args.deter          = defaults["deter"]     # GRU 确定性状态 dim
    p_args.hidden         = defaults["hidden"]    # 头/解码 MLP 隐层
    p_args.blocks         = defaults["blocks"]    # BlockLinear 分块数
    p_args.stoch_groups   = defaults["stoch_groups"]
    p_args.stoch_classes  = defaults["stoch_classes"]
    p_args.action_embed   = 64                    # 离散动作 embedding 维度
    p_args.use_graph_pool = False                 # 若你希望复用邻接池化，可开 True

    # KL 与分布（DreamerV3 关键）
    p_args.kl_dyn_scale   = 1.0
    p_args.kl_rep_scale   = 1.0
    p_args.kl_free_nats   = 1.0
    p_args.unimix         = 0.01                  # unimix ε

    # 重建/奖励/折扣损失权重（贴近官方）
    p_args.obs_loss_scale     = 1.0
    p_args.rew_loss_scale     = 3.5
    p_args.done_loss_scale    = 1.0

    # 奖励 two-hot 量化
    p_args.rew_bins       = 255
    p_args.rew_min        = -1000.0
    p_args.rew_max        = 1000.0

    # 优化细节
    p_args.lr             = 3e-4
    p_args.grad_clip      = 100.0

    # —— 图像分支（与 DreamerV3 对齐，可随时关闭 use_images）——
    p_args.use_images         = True             # 未来环境侧引入图像可直接训练
    p_args.img_channels       = 3
    p_args.img_height         = 84
    p_args.img_width          = 84
    p_args.enc_img_dim        = defaults["enc_img_dim"]
    p_args.obs_img_loss_scale = 1.0

    # 邻接矩阵（若用图池化）
    agent_args.adj              = _adj_from_env(env, n_agent)

    # 绑定半径与网络
    agent_args.radius_v         = HP.radius_v
    agent_args.radius_pi        = HP.radius_pi
    agent_args.pi_args          = pi_args
    agent_args.v_args           = v_args
    agent_args.p_args           = p_args

    # 顶层也挂上网络与超参（有的 Logger/Runner 会直接读）
    alg.pi_args          = pi_args
    alg.v_args           = v_args
    alg.agent_args       = agent_args
    return alg
