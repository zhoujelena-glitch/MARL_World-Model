import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
#from algorithms.algorithm import ReplayBuffer
#from ray.state import actors
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor, EnsembledModel, SquashedGaussianActor, ParameterizedModel_MBPPO
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse
from typing import Any, Optional

from algorithms.algo.buffer import MultiCollect,Trajectory,TrajectoryBuffer,ModelBuffer
from algorithms.algo.normalization_utils import ZFilter, RunningStat


def _resolve_schedule(sched, default, it):
    """sched 可以是 callable(it)->int，或 [(step, val), ...] 的列表；default 是没有 sched 时的回退值。"""
    if sched is None:
        return int(default)
    if callable(sched):
        return int(sched(it))
    # 支持列表 [(step,val), ...]
    try:
        val = default
        for step, v in sched:
            if it >= step:
                val = v
            else:
                break
        return int(val)
    except Exception:
        return int(default)


def translate_action(action_bias, action_scale, action):
    action = torch.as_tensor(action, dtype=torch.float)
    actions = action.detach().squeeze()
    cp_actions = torch.clamp(actions, min=-1.0, max=1.0)
    low = action_bias - action_scale
    high = action_bias + action_scale
    cp_actions = 0.5 * (cp_actions + 1.0) * (high - low) + low
    cp_actions = cp_actions.cpu().numpy()
    return cp_actions

def translate_action_2(action_bias, action_scale, action):
    actions = action.detach().squeeze()
    cp_actions = torch.clamp(actions, min=-1.0, max=1.0)
    low = action_bias - action_scale
    high = action_bias + action_scale
    cp_actions = 0.5 * (cp_actions + 1.0) * (high - low) + low
    return cp_actions


def transfer_action_real_power(a,result):
    b = np.array([])
    for i in range(result.shape[0]):
        row = a[i, :]
        mask = result[i, :]
        non_zeros = row[np.nonzero(mask)]
        res = a[i, len(non_zeros):]
        if len(res)!=0:
            b = np.concatenate([b,np.mean(res) + non_zeros])
        else:
            b = np.concatenate([b,non_zeros])
    return b


class OnPolicyRunner:
    def __init__(self, logger, run_args, alg_args, agent, env_learn, env_test, env_args, **kwargs):
        self.logger = logger
        self.name = run_args.name
        if not run_args.init_checkpoint is None:
            agent.load(run_args.init_checkpoint)
            logger.log(interaction=run_args.start_step)
        self.start_step = run_args.start_step
        self.env_name = env_args.env
        self.algo_name = env_args.algo

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_inner_iter = alg_args.n_inner_iter
        self.n_warmup = alg_args.n_warmup
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.test_length = alg_args.test_length
        self.max_episode_len = alg_args.max_episode_len
        self.clip_scheme = None if (not hasattr(alg_args, "clip_scheme")) else alg_args.clip_scheme

        # agent / device
        self.agent = agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        # envs
        self.env_learn = env_learn
        self.env_test = env_test
        if self.env_name == 'PowerGrid' and self.env_learn.n_agent==40:
            self.running_state = ZFilter((self.env_learn.n_agent,self.env_learn.n_s), clip=5.0)
        if self.env_name == 'Large_city':
            self.running_state = ZFilter((self.env_learn.n_agent,self.env_learn.n_s), clip=5.0)

        # buffers / model-based flags
        self.discrete = agent.discrete
        action_dtype = torch.long if self.discrete else torch.float
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.n_traj = alg_args.n_traj
            self.model_traj_length = alg_args.model_traj_length
            self.model_error_thres = alg_args.model_error_thres
            self.model_buffer = ModelBuffer(alg_args.model_buffer_size)
            self.model_update_length = alg_args.model_update_length
            self.model_validate_interval = alg_args.model_validate_interval
            self.model_length_schedule = alg_args.model_length_schedule
            self.model_prob = alg_args.model_prob
        self.s, self.episode_len, self.episode_reward = self.env_learn.reset(), 0, 0

        # pretrained world model
        self.load_pretrained_model = alg_args.load_pretrained_model
        if self.model_based and self.load_pretrained_model:
            self.agent.load_model(alg_args.pretrained_model)

        if self.env_name == 'Real_Power':
            self.real_power_action_meam = (np.array([self.env_test.action_space.low]*self.env_test.n_agents) + np.array([self.env_test.action_space.high]*self.env_test.n_agents))/2
            self.real_power_action_var = (np.array([self.env_test.action_space.high]*self.env_test.n_agents) - np.array([self.env_test.action_space.low]*self.env_test.n_agents))/2
            self.running_state = ZFilter((self.env_learn.n_agents,self.env_learn.obs_size), clip=5.0)
        elif self.env_name == 'Pandemic':
            self.running_state = ZFilter((self.env_learn.n_agent,self.env_learn.n_s), clip=5.0)
            s_min = np.array([[0]*16]*10)
            s_max = []
            num_persons = 500
            for i in range(len(self.env_learn.Nums_Location)):
                s_max.append(np.concatenate((np.array([self.env_learn.Nums_Location[i]]*3), np.array([num_persons, num_persons, num_persons, num_persons, num_persons, num_persons, num_persons, num_persons, num_persons, num_persons, 4, 1, 120]))))
            s_max.append(np.array([1,1,1,num_persons,num_persons,num_persons,num_persons,num_persons,num_persons,num_persons,num_persons,num_persons,num_persons,4,1,120]))
            s_max = np.array(s_max)
            self.s_mean = (s_max + s_min)/2
            self.s_std = (s_max - s_min)/2

        # schedules
        self.model_length_schedule = alg_args.model_length_schedule
        self.model_update_length_schedule = getattr(alg_args, 'model_update_length_schedule', None)
        self.model_prob = alg_args.model_prob


    def run(self):
        # ===================== WARM-UP（仅世界模型） =====================
        if self.model_based and not self.load_pretrained_model:
            env = getattr(self, "env_learn", None)
            old_ban = None
            old_grace = None
            old_decint = None
            if env is not None:
                if hasattr(env, "ban_idle_when_queue"):
                    old_ban = env.ban_idle_when_queue
                    env.ban_idle_when_queue = True
                if hasattr(env, "idle_grace"):
                    old_grace = getattr(env, "idle_grace", 0)
                    env.idle_grace = 0
                if hasattr(env, "decision_interval"):
                    old_decint = getattr(env, "decision_interval", None)
                    env.decision_interval = 1
            try:
                for _ in trange(self.n_warmup, desc="warm-up collect"):
                    trajs = self.rollout_env()
                    self.model_buffer.storeTrajs(trajs)
                self.updateModel(self.n_model_update_warmup)
            finally:
                if env is not None:
                    if old_ban is not None:
                        env.ban_idle_when_queue = old_ban
                    if old_grace is not None:
                        env.idle_grace = old_grace
                    if old_decint is not None:
                        env.decision_interval = old_decint

        # ================== WARM-UP 结束，进入主训练 ======================
        for iter in trange(self.n_iter):
            if (iter % int(self.test_interval)) == 0:
                mean_return = self.test(iter)

            if (iter % int(self.test_interval)) == 0 and iter != 0:
                self.agent.save_nets(f'./checkpoints/{self.name}', iter)

            trajs = self.rollout_env()
            t1 = time.time()

            if self.model_based:
                self.model_buffer.storeTrajs(trajs)
                if (iter % int(self.test_interval)) == 0:
                    if self.model_update_length_schedule is not None:
                        self.model_update_length = _resolve_schedule(
                            self.model_update_length_schedule, self.model_update_length, iter
                        )
                    self.updateModel()
            t2 = time.time()
            print('t=', t2 - t1)

            agentInfo = []
            real_trajs = trajs
            for inner in trange(self.n_inner_iter):
                if self.model_based:
                    use_model = np.random.uniform() < self.model_prob
                    if use_model:
                        if self.model_length_schedule is not None:
                            length = _resolve_schedule(self.model_length_schedule, self.model_traj_length, iter)
                            trajs = self.rollout_model(real_trajs, length)
                        else:
                            trajs = self.rollout_model(real_trajs)
                    else:
                        trajs = trajs

                if self.clip_scheme is not None:
                    info = self.agent.updateAgent(trajs, self.clip_scheme(iter))
                else:
                    info = self.agent.updateAgent(trajs)
                agentInfo.append(info)

                if self.agent.checkConverged(agentInfo):
                    break

            self.logger.log(inner_iter=inner + 1, iter=iter)


    def test(self, nnn):
        time_t = time.time()
        length = self.test_length
        returns, scaled, lengths, episodes = [], [], [], []
        for i in trange(self.n_test):
            episode = []
            env = self.env_test

            if self.env_name == 'eight' or self.env_name == 'ring':
                if i==0 and nnn == 0:
                    env.reset()
            elif self.env_name == "Large_city":
                env.clear()
                env.reset()
            else:
                env.reset()

            d, ep_ret, ep_len = np.array([False]), 0, 0

            while not(d.any() or (ep_len == length)):
                if self.env_name == 'PowerGrid' and env.n_agent==40:
                    s = env.get_state_()
                    s = self.running_state(s)
                elif self.env_name == "Pandemic":
                    s = env.get_state_()
                    s = (s - self.s_mean) / self.s_std
                elif self.env_name == 'Real_Power':
                    s = env.get_state_()
                    s = np.array(s)
                    s = self.running_state(s)
                elif self.env_name == 'Large_city':
                    s = env.get_state_()
                    s = self.running_state(s)
                else:
                    s = env.get_state_()

                s = torch.as_tensor(s, dtype=torch.float, device=self.device)
                a = self.agent.act(s, if_test=True).sample()
                a = a.detach().cpu().numpy()

                if (self.env_name == 'Monaco' and self.algo_name == 'IC3Net') or (self.env_name == 'Grid' and self.algo_name == 'IC3Net'):
                    s1, r, d, _ = env.step(np.squeeze(a))
                elif self.env_name == 'PowerGrid':
                    if self.algo_name == 'IA2C' or self.algo_name == 'IC3Net':
                        s1, r, d, _ = env.step(np.squeeze(a))
                    else:
                        s1, r, d, _ = env.step(a)
                    if env.n_agent==40:
                        s1 = self.running_state(s1)
                elif self.env_name == "Pandemic":
                    if self.algo_name == 'IA2C' or self.algo_name == 'IC3Net':
                        s1, r, d, _ = env.step(np.squeeze(a))
                        s1 = (s1 - self.s_mean) / self.s_std
                    else:
                        s1, r, d, _ = env.step(a)
                        s1 = (s1 - self.s_mean) / self.s_std
                elif self.env_name == 'Large_city':
                    if self.algo_name == 'IA2C' or self.algo_name == 'IC3Net':
                        s1, r, d, _ = env.step(np.squeeze(a))
                    else:
                        s1, r, d, _ = env.step(a)
                    s1 = self.running_state(s1)
                elif self.env_name == 'Real_Power':
                    r, d, info = env.step(a)
                    s1 = env.get_state_()
                    s1 = np.array(s1)
                    s1 = self.running_state(s1)
                    r = np.array([info["totally_controllable_ratio"]]*env.n_agents)
                    d = [d]*env.n_agents
                else:
                    s1, r, d, _ = env.step(a)

                episode += [(s.tolist(), a.tolist(), r.tolist())]
                d = np.array(d)
                ep_ret += r.sum()
                ep_len += 1
                self.logger.log(interaction=None)

            if self.env_name != "Large_city":
                if hasattr(env, 'rescaleReward'):
                    scaled += [ep_ret]
                    ep_ret = env.rescaleReward(ep_ret, ep_len)
            returns += [ep_ret]
            lengths += [ep_len]
            episodes += [episode]

        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)

        self.logger.log(test_episode_reward=returns, test_episode_len=lengths, test_round=None)
        print(returns)
        print(f"{self.n_test} episodes average accumulated reward: {returns.mean()}")
        if self.env_name != "Large_city":
            if hasattr(env, 'rescaleReward'):
                print(f"scaled reward {np.mean(scaled)}")
        with open(f"checkpoints/{self.name}/test.pickle", "wb") as f:
            pickle.dump(episodes, f)
        with open(f"checkpoints/{self.name}/test.txt", "w") as f:
            for episode in episodes:
                for step in episode:
                    f.write(f"{step[0]}, {step[1]}, {step[2]}\n")
                f.write("\n")
        self.logger.log(test_time=time.time()-time_t)
        return returns.mean()

    def rollout_env(self, length = 0):
        """收集真实环境轨迹；若 env 在 info 里返回 images_uint8，则一并写入 TrajectoryBuffer。"""
        time_t = time.time()
        if length <= 0:
            length = self.rollout_length
        env = self.env_learn
        trajs = []
        traj = TrajectoryBuffer(device=self.device)
        start = time.time()

        if self.env_name == 'Real_Power':
            totally_controllable_ratio = 0

        if self.env_name in ('catchup','slowdown','Grid','Monaco'):
            env.reset()

        for t in range(length):
            s = env.get_state_()

            if self.env_name == 'PowerGrid' and env.n_agent==40:
                s = self.running_state(s)
            elif self.env_name == "Pandemic":
                s = (s - self.s_mean) / self.s_std
            elif self.env_name == 'Real_Power':
                s = np.array(s)
                s = self.running_state(s)
            elif self.env_name == 'Large_city':
                s = self.running_state(s)

            s = torch.as_tensor(s, dtype=torch.float, device=self.device)
            dist = self.agent.act(s)
            a = dist.sample()
            logp = dist.log_prob(a)
            a_np = a.detach().cpu().numpy()

            # === 环境步进，接住 info ===
            info: Optional[dict] = None
            if (self.env_name == 'Monaco' and self.algo_name == 'IC3Net') or (self.env_name == 'Grid' and self.algo_name == 'IC3Net'):
                s1, r, d, info = env.step(np.squeeze(a_np))
            elif self.env_name == 'PowerGrid':
                if self.algo_name in ('IA2C','IC3Net'):
                    s1, r, d, info = env.step(np.squeeze(a_np))
                    d = np.array([d]*env.n_agent)
                else:
                    s1, r, d, info = env.step(a_np)
                    d = np.array([d]*env.n_agent)
                if env.n_agent==40:
                    s1 = self.running_state(s1)
            elif self.env_name == "Pandemic":
                if self.algo_name in ('IA2C','IC3Net'):
                    s1, r, d, info = env.step(np.squeeze(a_np))
                    s1 = (s1 - self.s_mean) / self.s_std
                else:
                    s1, r, d, info = env.step(a_np)
                    s1 = (s1 - self.s_mean) / self.s_std
            elif self.env_name == 'Large_city':
                if self.algo_name in ('IA2C','IC3Net'):
                    s1, r, d, info = env.step(np.squeeze(a_np))
                else:
                    s1, r, d, info = env.step(a_np)
                s1 = self.running_state(s1)
            elif self.env_name == 'Real_Power':
                r, d, info = env.step(a_np)
                s1 = env.get_state_()
                s1 = np.array(s1)
                s1 = self.running_state(s1)
                r = np.array([info["totally_controllable_ratio"]]*env.n_agents, dtype=np.float32)
                d = np.array([d]*env.n_agents)
                totally_controllable_ratio += info.get("totally_controllable_ratio", 0.0)
            else:
                s1, r, d, info = env.step(a_np)

            # --- 从 info 里取图像（若有） ---
            img = None
            if isinstance(info, dict) and ("images_uint8" in info):
                img_np = info["images_uint8"]  # 期望 [N,C,H,W] uint8
                try:
                    if isinstance(img_np, np.ndarray):
                        img = torch.from_numpy(img_np).to(self.device, non_blocking=True)
                    else:
                        # 兼容 list/torch
                        img = torch.as_tensor(img_np, device=self.device)
                    if img.dtype != torch.uint8:
                        img = img.to(torch.uint8)
                except Exception:
                    img = None

            # —— 存入 buffer：若你的 TrajectoryBuffer.store 没有 img 参数，会自动回退 ——
            try:
                traj.store(s, a, r, s1, d, logp, img=img)
            except TypeError:
                traj.store(s, a, r, s1, d, logp)

            episode_r = r
            if hasattr(env, '_comparable_reward'):
                episode_r = env._comparable_reward()
            if getattr(episode_r, "ndim", 1) > 1:
                episode_r = episode_r.mean(axis=0)

            self.episode_reward += episode_r
            self.episode_len += 1
            self.logger.log(interaction=None)
            if self.episode_len == self.max_episode_len:
                d = np.zeros(np.array(d).shape, dtype=np.float32)
            d = np.array(d)

            # ====== 各环境的 episode 收尾 ======
            if self.env_name in ('catchup','slowdown'):
                if (self.env_name == 'catchup' and self.episode_len == self.max_episode_len) or \
                   (self.env_name == 'slowdown' and (d.any() or self.episode_len == self.max_episode_len)):
                    self.logger.log(episode_reward=self.episode_reward.sum()/600, episode_len=self.episode_len, episode=None)
                    try:
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                    if self.episode_len == self.max_episode_len and self.model_based:
                        trajs += traj.retrieve()
                        traj = TrajectoryBuffer(device=self.device)

            elif self.env_name in ('eight','ring'):
                if self.episode_len == self.max_episode_len:
                    self.logger.log(episode_reward=self.episode_reward.sum(), episode_len=self.episode_len, episode=None)
                    try:
                        self.episode_reward, self.episode_len = 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        self.episode_reward, self.episode_len = 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                    if self.episode_len == self.max_episode_len and self.model_based:
                        trajs += traj.retrieve()
                        traj = TrajectoryBuffer(device=self.device)

            elif self.env_name == 'PowerGrid':
                if d.any() or (self.episode_len == self.max_episode_len):
                    self.logger.log(episode_reward=self.episode_reward.sum()/env.T, episode_len=self.episode_len, episode=None)
                    try:
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                if self.episode_len == self.max_episode_len and self.model_based:
                    trajs += traj.retrieve()
                    traj = TrajectoryBuffer(device=self.device)

            elif self.env_name == 'Real_Power':
                if d.any() or (self.episode_len == self.max_episode_len):
                    self.logger.log(episode_reward=self.episode_reward.sum(), episode_len=self.episode_len, episode=None, totally_controllable_ratio=totally_controllable_ratio)
                    try:
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                if self.episode_len == self.max_episode_len and self.model_based:
                    trajs += traj.retrieve()
                    traj = TrajectoryBuffer(device=self.device)

            elif self.env_name == 'Large_city':
                if self.episode_len == self.max_episode_len:
                    self.logger.log(episode_reward=self.episode_reward.sum(), episode_len=self.episode_len, episode=None)
                    try:
                        self.env_learn.clear()
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        self.env_learn.clear()
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                if self.episode_len == self.max_episode_len and self.model_based:
                    trajs += traj.retrieve()
                    traj = TrajectoryBuffer(device=self.device)

            else:
                if d.any() or (self.episode_len == self.max_episode_len):
                    self.logger.log(episode_reward=self.episode_reward.sum(), episode_len=self.episode_len, episode=None)
                    try:
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    except Exception as e:
                        print('reset error!:', e)
                        _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                        if not self.model_based:
                            trajs += traj.retrieve()
                            traj = TrajectoryBuffer(device=self.device)
                if self.episode_len == self.max_episode_len and self.model_based:
                    trajs += traj.retrieve()
                    traj = TrajectoryBuffer(device=self.device)

        end = time.time()
        print('time in 1 episode is ', end - start)
        trajs += traj.retrieve(length=self.max_episode_len)
        self.logger.log(env_rollout_time=time.time()-time_t)
        return trajs

    def rollout_model(self, trajs, length=0):
        time_t = time.time()
        n_traj = self.n_traj
        if length <= 0:
            length = self.model_traj_length
        s = [traj['s'] for traj in trajs]
        s = torch.stack(s, dim=0)
        b, T, n, depth = s.shape
        s = s.view(-1, n, depth)
        idxs = torch.randint(low=0, high=b * T, size=(n_traj,), device=self.device)
        s = s.index_select(dim=0, index=idxs)

        trajs = TrajectoryBuffer(device=self.device)
        for _ in range(length):
            dist = self.agent.act(s)
            a = dist.sample()
            logp = dist.log_prob(a)
            r, s1, d, _ = self.agent.model_step(s, a)

            if self.env_name == 'UAV_9d':
                env = self.env_learn
                s = env.get_model_state(s,self.device)
                s1 = env.get_model_state(s1,self.device)
                r = env.get_model_reward(s1,self.device)

            try:
                trajs.store(s, a, r, s1, d, logp)  # 模型 rollout 不含相机帧
            except TypeError:
                trajs.store(s, a, r, s1, d, logp)
            s = s1
        trajs = trajs.retrieve()
        self.logger.log(model_rollout_time=time.time()-time_t)
        return trajs

    def updateModel(self, n=0):
        if n <= 0:
            n = self.n_model_update
        for i_model_update in trange(n):
            trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
            trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]
            self.agent.updateModel(trajs, length=self.model_update_length)

            if i_model_update % self.model_validate_interval == 0:
                validate_trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
                validate_trajs = [traj.getFraction(length=self.model_update_length) for traj in validate_trajs]
                rel_error = self.agent.validateModel(validate_trajs, length=self.model_update_length)
                if rel_error < self.model_error_thres:
                    break
        self.logger.log(model_update = i_model_update + 1)

    def testModel(self, n = 0):
        trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
        trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]
        return self.agent.validateModel(trajs, length=self.model_update_length)
