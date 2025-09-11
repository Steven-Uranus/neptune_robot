# 完整训练流程函数调用总结

本文档汇总了HumanoidVerse框架中一次完整训练的所有关键函数调用，按执行顺序详细列出每个文件中的具体函数。

## 🚀 1. 程序启动 (train_agent.py)

```python
train_agent.py::main(config)
├── 1. 配置和环境设置
│   ├── simulator_type = config.simulator['_target_'].split('.')[-1]
│   ├── 设置日志系统: logger.add(hydra_log_path)
│   ├── 初始化WandB: wandb.init() [可选]
│   ├── 设置设备: device = "cuda:0" if torch.cuda.is_available() else "cpu"
│   └── pre_process_config(config)
├── 2. 实例化组件
│   ├── env = instantiate(config.env, device=device)
│   └── algo = instantiate(device=device, env=env, config=config.algo)
└── 3. 开始训练
    ├── algo.setup()
    ├── algo.load(checkpoint) [可选]
    └── algo.learn()  # 🎯 进入主训练循环
```

## 🌍 2. 环境初始化 (BaseTask → LeggedRobotBase → MotionTracking)

```python
# BaseTask.__init__()
BaseTask.__init__(config, device)
├── torch._C._jit_set_profiling_mode(False)
├── SimulatorClass = get_class(config.simulator._target_)
├── self.simulator = SimulatorClass(config, device)
├── self.simulator.set_headless(headless)
├── self.simulator.setup()
├── self.simulator.setup_terrain(terrain_mesh_type)
├── self._load_assets()
├── self._get_env_origins()
├── self._create_envs()
├── self.simulator.prepare_sim()
├── self.simulator.setup_viewer() [如果非headless]
└── self._init_buffers()

# LeggedRobotBase.__init__()
LeggedRobotBase.__init__(config, device)
├── super().__init__(config, device)  # 调用BaseTask
├── self._domain_rand_config()
├── self._prepare_reward_function()
└── self.history_handler = HistoryHandler(...)

# MotionTracking.__init__()
LeggedRobotMotionTracking.__init__(config, device)
├── super().__init__(config, device)  # 调用LeggedRobotBase
├── self._init_motion_lib()
├── self._init_motion_extend()
├── self._init_tracking_config()
└── self._init_save_motion()
```

## 🧠 3. 算法初始化 (PPO)

```python
# PPO.__init__()
PPO.__init__(env, config, log_dir, device)
├── self.env = env
├── self.config = config
├── self.device = device
├── self._init_config()
├── 初始化统计变量 (rewbuffer, lenbuffer等)
└── self.env.reset_all()

# PPO.setup()
PPO.setup()
├── self._setup_models_and_optimizer()
│   ├── self.actor = PPOActor(obs_dim_dict, config, num_actions, init_noise_std)
│   ├── self.critic = PPOCritic(obs_dim_dict, config)
│   ├── self.actor_optimizer = Adam(self.actor.parameters(), lr)
│   └── self.critic_optimizer = Adam(self.critic.parameters(), lr)
└── self._setup_storage()
    ├── self.storage = RolloutStorage(num_envs, num_steps_per_env)
    ├── 注册观测键值 (actor_obs, critic_obs等)
    └── 注册策略键值 (actions, rewards, values等)
```

## 🎯 4. 主训练循环 (PPO.learn)

```python
PPO.learn()
├── 初始化设置
│   ├── 设置随机episode长度 [可选]
│   ├── obs_dict = self.env.reset_all()
│   ├── obs_dict[obs_key].to(self.device)
│   └── self._train_mode()
└── for it in range(current_iter, tot_iter):  # 主训练循环
    ├── start_time = time.time()
    ├── obs_dict = self._rollout_step(obs_dict)     # 🔄 数据收集
    ├── loss_dict = self._training_step()           # 📈 策略更新
    ├── stop_time = time.time()
    ├── self._post_epoch_logging(log_dict)          # 📊 日志记录
    ├── self.save() [按间隔]                        # 💾 保存模型
    └── self.ep_infos.clear()
```

## 🔄 5. 数据收集阶段 (PPO._rollout_step)

```python
PPO._rollout_step(obs_dict)
├── with torch.inference_mode():
└── for i in range(self.num_steps_per_env):
    ├── 🎭 策略推理
    │   ├── policy_state_dict = self._actor_rollout_step(obs_dict, {})
    │   │   ├── actions = self._actor_act_step(obs_dict)
    │   │   │   └── self.actor.act(obs_dict["actor_obs"])
    │   │   │       ├── self.actor.update_distribution(actor_obs)
    │   │   │       └── self.distribution.sample()
    │   │   ├── action_mean = self.actor.action_mean.detach()
    │   │   ├── actions_log_prob = self.actor.get_actions_log_prob(actions)
    │   │   └── return policy_state_dict
    │   └── values = self._critic_eval_step(obs_dict)
    │       └── self.critic.evaluate(obs_dict["critic_obs"])
    ├── 💾 存储到缓冲区
    │   ├── self.storage.update_key(obs_key, obs_dict[obs_key])
    │   └── self.storage.update_key(key, policy_state_dict[key])
    ├── 🌍 环境交互
    │   └── obs_dict, rewards, dones, infos = self.env.step({"actions": actions})
    │       └── 调用环境step (见6.环境Step)
    ├── 📊 记录和处理
    │   ├── self.storage.update_key('rewards', rewards)
    │   ├── self.storage.update_key('dones', dones)
    │   ├── self.storage.increment_step()
    │   └── self._process_env_step(rewards, dones, infos)
    └── 📈 统计更新
        ├── self.ep_infos.append(infos['episode'])
        ├── self.cur_reward_sum += rewards
        └── 更新episode缓冲区
└── 🧮 计算GAE
    ├── returns, advantages = self._compute_returns(obs_dict, policy_state_dict)
    ├── self.storage.batch_update_data('returns', returns)
    └── self.storage.batch_update_data('advantages', advantages)
```

## 🌍 6. 环境Step (LeggedRobotBase.step)

```python
LeggedRobotBase.step(actor_state)
├── actions = actor_state["actions"]
├── self._pre_physics_step(actions)
│   ├── self.actions = torch.clip(actions, -clip_limit, clip_limit)
│   └── 控制延迟处理 [如果启用]
├── self._physics_step()
│   ├── self.render()
│   └── for _ in range(control_decimation):
│       ├── self._apply_force_in_physics_step()
│       │   ├── self.torques = self._compute_torques(self.actions_after_delay)
│       │   └── self.simulator.apply_torques_at_dof(self.torques)
│       └── self.simulator.simulate_at_each_physics_step()
└── self._post_physics_step()
    ├── self._refresh_sim_tensors()
    ├── self.episode_length_buf += 1
    ├── self._pre_compute_observations_callback()
    │   └── 对于MotionTracking: 获取参考动作状态
    ├── self._check_termination()
    ├── self._compute_reward()
    ├── self.reset_envs_idx(env_ids)
    │   ├── self._reset_root_states(env_ids)
    │   └── self._reset_dofs(env_ids)
    ├── self._compute_observations()
    ├── self._post_compute_observations_callback()
    └── torch.clip(obs_val, -clip_obs, clip_obs)
```

## 📈 7. 策略更新阶段 (PPO._training_step)

```python
PPO._training_step()
├── loss_dict = self._init_loss_dict_at_training_step()
├── generator = self.storage.mini_batch_generator(num_mini_batches, num_epochs)
├── for policy_state_dict in generator:
│   ├── policy_state_dict[key].to(self.device)
│   └── loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
│       └── loss_dict = self._update_ppo(policy_state_dict, loss_dict)
│           ├── 提取batch数据
│           ├── 重新计算策略输出
│           │   ├── self._actor_act_step(policy_state_dict)
│           │   ├── actions_log_prob = self.actor.get_actions_log_prob(actions)
│           │   ├── value_batch = self._critic_eval_step(policy_state_dict)
│           │   └── entropy_batch = self.actor.entropy
│           ├── 自适应学习率调整 [可选]
│           ├── 计算PPO损失
│           │   ├── ratio = exp(new_log_prob - old_log_prob)
│           │   ├── surrogate_loss = max(surrogate, surrogate_clipped).mean()
│           │   ├── value_loss = (returns - values)².mean()
│           │   └── entropy_loss = entropy.mean()
│           ├── 反向传播和优化
│           │   ├── self.actor_optimizer.zero_grad()
│           │   ├── self.critic_optimizer.zero_grad()
│           │   ├── actor_loss.backward()
│           │   ├── critic_loss.backward()
│           │   ├── nn.utils.clip_grad_norm_(parameters, max_grad_norm)
│           │   ├── self.actor_optimizer.step()
│           │   └── self.critic_optimizer.step()
│           └── 更新损失字典
├── 计算平均损失
└── self.storage.clear()
```

## 🎯 8. Motion Tracking特殊处理

```python
# Motion Tracking在环境step中的特殊处理
LeggedRobotMotionTracking._pre_compute_observations_callback()
├── super()._pre_compute_observations_callback()
├── motion_times = (episode_length_buf + 1) * dt + motion_start_times
├── motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset)
└── 存储参考状态 (ref_root_pos, ref_root_rot, ref_dof_pos等)

# Motion Tracking重置
LeggedRobotMotionTracking._reset_root_states(env_ids)
├── motion_res = self._motion_lib.get_motion_state(...)
├── 添加初始化噪声
└── 设置机器人状态为参考状态+噪声

LeggedRobotMotionTracking._reset_dofs(env_ids)
├── motion_res = self._motion_lib.get_motion_state(...)
├── 添加关节噪声
└── 设置关节状态
```

## 🔄 9. 完整的一次训练迭代流程

```
1. PPO.learn() 开始一次迭代
2. PPO._rollout_step() 数据收集
   ├── for step in num_steps_per_env:
   │   ├── Actor推理 → 生成动作
   │   ├── Critic推理 → 估计价值
   │   ├── Environment.step() → 环境交互
   │   │   ├── 预处理 → 动作裁剪、延迟
   │   │   ├── 物理仿真 → 计算扭矩、仿真步进
   │   │   └── 后处理 → 奖励、观测、重置
   │   └── 存储经验到RolloutStorage
   └── 计算GAE优势函数和回报
3. PPO._training_step() 策略更新
   ├── for mini_batch in generator:
   │   ├── 重新计算策略输出
   │   ├── 计算PPO三种损失
   │   └── 反向传播和优化
   └── 清空存储器
4. 日志记录和模型保存
5. 进入下一次迭代
```

这个完整的函数调用总结展示了从程序启动到一次训练迭代完成的所有关键函数调用，帮助您理解整个训练流程的执行顺序和各组件之间的交互关系。
