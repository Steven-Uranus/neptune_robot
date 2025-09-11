# HumanoidVerse训练流程完整调用顺序

本文档详细列出了HumanoidVerse框架中一次完整训练流程的代码调用顺序，从启动到一次训练迭代的所有函数调用。

## 🚀 1. 程序启动阶段

### 1.1 主函数入口
```
📁 humanoidverse/train_agent.py
├── main(config: OmegaConf)                    # Hydra主函数入口
│   ├── 检测仿真器类型 (IsaacGym/IsaacSim/Genesis)
│   ├── 设置日志系统
│   ├── 初始化WandB (如果启用)
│   ├── 设置设备 (CPU/CUDA)
│   ├── pre_process_config(config)             # 预处理配置
│   │
│   ├── env = instantiate(config.env)          # 🌍 实例化环境
│   │   └── 调用环境构造函数 (见2.环境初始化)
│   │
│   ├── algo = instantiate(config.algo)        # 🧠 实例化算法
│   │   └── 调用算法构造函数 (见3.算法初始化)
│   │
│   ├── algo.setup()                           # 设置算法组件
│   ├── algo.load(checkpoint) [可选]           # 加载检查点
│   └── algo.learn()                           # 🎯 开始训练 (见4.训练循环)
```

## 🌍 2. 环境初始化阶段

### 2.1 环境类继承链
```
BaseTask.__init__()
├── 创建仿真器实例
├── 设置仿真参数
├── 加载机器人资产
├── 创建环境
└── 初始化缓冲区
    ↓
LeggedRobotBase.__init__()
├── super().__init__()                         # 调用BaseTask初始化
├── _domain_rand_config()                      # 域随机化配置
├── _prepare_reward_function()                 # 准备奖励函数
└── 初始化历史观测处理器
    ↓
LeggedRobotMotionTracking.__init__()
├── super().__init__()                         # 调用LeggedRobotBase初始化
├── _init_motion_lib()                         # 初始化动作库
├── _init_motion_extend()                      # 初始化动作扩展
├── _init_tracking_config()                    # 初始化追踪配置
└── _init_save_motion()                        # 初始化动作保存
```

### 2.2 BaseTask详细初始化
```
📁 humanoidverse/envs/base_task/base_task.py
BaseTask.__init__(config, device)
├── 设置PyTorch JIT优化
├── 创建仿真器
│   ├── SimulatorClass = get_class(config.simulator._target_)
│   ├── self.simulator = SimulatorClass(config, device)
│   ├── self.simulator.set_headless(headless)
│   └── self.simulator.setup()
├── 设置基础参数 (dt, num_envs, 观测/动作维度)
├── self.simulator.setup_terrain()
├── self._load_assets()                        # 加载机器人资产
├── self._get_env_origins()                    # 获取环境原点
├── self._create_envs()                        # 创建环境
├── self.simulator.prepare_sim()               # 准备仿真
├── self.simulator.setup_viewer() [如果非headless]
└── self._init_buffers()                       # 初始化缓冲区
```

## 🧠 3. 算法初始化阶段

### 3.1 PPO算法初始化
```
📁 humanoidverse/agents/ppo/ppo.py
PPO.__init__(env, config, log_dir, device)
├── 保存环境和配置引用
├── 创建TensorBoard写入器
├── self._init_config()                        # 初始化配置参数
├── 初始化统计变量 (rewbuffer, lenbuffer等)
├── 创建评估回调列表
└── self.env.reset_all()                       # 重置所有环境
```

### 3.2 PPO.setup()详细流程
```
PPO.setup()
├── self._setup_models_and_optimizer()
│   ├── 创建PPOActor网络
│   │   ├── self.actor = PPOActor(obs_dim_dict, config, num_actions, init_noise_std)
│   │   └── self.actor.to(device)
│   ├── 创建PPOCritic网络
│   │   ├── self.critic = PPOCritic(obs_dim_dict, config)
│   │   └── self.critic.to(device)
│   ├── 创建Actor优化器: Adam(actor.parameters(), lr)
│   └── 创建Critic优化器: Adam(critic.parameters(), lr)
└── self._setup_storage()
    ├── 创建RolloutStorage(num_envs, num_steps_per_env)
    ├── 注册观测键值 (actor_obs, critic_obs等)
    └── 注册策略键值 (actions, rewards, values等)
```

## 🎯 4. 主训练循环

### 4.1 PPO.learn()主循环
```
📁 humanoidverse/agents/ppo/ppo.py
PPO.learn()
├── 随机初始化episode长度 [可选]
├── obs_dict = self.env.reset_all()            # 重置所有环境
├── self._train_mode()                         # 设置训练模式
└── for it in range(num_learning_iterations):  # 主训练循环
    ├── obs_dict = self._rollout_step(obs_dict)    # 🔄 数据收集阶段
    ├── loss_dict = self._training_step()          # 📈 策略更新阶段
    ├── 记录日志和统计信息
    └── 保存模型 [按间隔]
```

### 4.2 数据收集阶段 (_rollout_step)
```
PPO._rollout_step(obs_dict)
├── with torch.inference_mode():               # 推理模式，不计算梯度
└── for i in range(num_steps_per_env):         # 每个环境收集N步
    ├── 🎭 策略推理阶段
    │   ├── policy_state_dict = self._actor_rollout_step(obs_dict, {})
    │   │   ├── actions = self._actor_act_step(obs_dict)
    │   │   │   └── return self.actor.act(obs_dict["actor_obs"])
    │   │   │       ├── self.actor.update_distribution(actor_obs)
    │   │   │       │   ├── mean = self.actor_module(actor_obs)
    │   │   │       │   └── self.distribution = Normal(mean, std)
    │   │   │       └── return self.distribution.sample()
    │   │   ├── action_mean = self.actor.action_mean.detach()
    │   │   ├── action_sigma = self.actor.action_std.detach()
    │   │   └── actions_log_prob = self.actor.get_actions_log_prob(actions)
    │   └── values = self._critic_eval_step(obs_dict)
    │       └── return self.critic.evaluate(obs_dict["critic_obs"])
    │
    ├── 💾 存储状态到缓冲区
    │   ├── 存储观测: storage.update_key(obs_key, obs_dict[obs_key])
    │   └── 存储策略状态: storage.update_key(key, policy_state_dict[key])
    │
    ├── 🌍 环境交互阶段
    │   ├── actor_state = {"actions": actions}
    │   └── obs_dict, rewards, dones, infos = self.env.step(actor_state)
    │       └── 调用环境step函数 (见5.环境Step详细流程)
    │
    ├── 📊 记录和处理
    │   ├── 转移数据到设备
    │   ├── 处理timeout奖励
    │   ├── storage.update_key('rewards', rewards)
    │   ├── storage.update_key('dones', dones)
    │   ├── storage.increment_step()
    │   └── self._process_env_step(rewards, dones, infos)
    │       ├── self.actor.reset(dones)
    │       └── self.critic.reset(dones)
    │
    └── 📈 统计信息更新
        ├── 更新episode奖励和长度
        └── 记录完成的episode到缓冲区
│
└── 🧮 计算GAE和回报
    ├── returns, advantages = self._compute_returns(obs_dict, policy_state_dict)
    │   ├── last_values = self.critic.evaluate(last_obs_dict["critic_obs"])
    │   └── for step in reversed(range(num_steps)):
    │       ├── delta = rewards[step] + γ*next_values - values[step]
    │       ├── advantage = delta + γ*λ*advantage
    │       └── returns[step] = advantage + values[step]
    ├── storage.batch_update_data('returns', returns)
    └── storage.batch_update_data('advantages', advantages)
```

## 🌍 5. 环境Step详细流程

### 5.1 环境step主流程
```
📁 humanoidverse/envs/legged_base_task/legged_robot_base.py
LeggedRobotBase.step(actor_state)
├── actions = actor_state["actions"]
├── self._pre_physics_step(actions)            # 🔧 预处理阶段
├── self._physics_step()                       # ⚡ 物理仿真阶段
├── self._post_physics_step()                  # 🔄 后处理阶段
└── return obs_buf_dict, rew_buf, reset_buf, extras
```

### 5.2 预处理阶段 (_pre_physics_step)
```
LeggedRobotBase._pre_physics_step(actions)
├── 🔒 动作裁剪
│   ├── clip_limit = config.robot.control.action_clip_value
│   └── self.actions = torch.clip(actions, -clip_limit, clip_limit)
├── 📊 记录裁剪统计
└── ⏱️ 控制延迟处理 [如果启用域随机化]
    ├── self.action_queue[:, 1:] = self.action_queue[:, :-1]
    ├── self.action_queue[:, 0] = self.actions
    └── self.actions_after_delay = self.action_queue[env_ids, delay_idx]
```

### 5.3 物理仿真阶段 (_physics_step)
```
LeggedRobotBase._physics_step()
├── self.render()                              # 渲染 [如果启用]
└── for _ in range(control_decimation):        # 控制抽取循环
    ├── self._apply_force_in_physics_step()    # 应用力/扭矩
    │   └── self.simulator.set_dof_torque_tensor(torques)
    └── self.simulator.simulate_at_each_physics_step()  # 执行物理步
```

### 5.4 后处理阶段 (_post_physics_step)
```
LeggedRobotBase._post_physics_step()
├── 🔄 刷新仿真状态
│   └── self._refresh_sim_tensors()
│       └── self.simulator.refresh_sim_tensors()
├── 📊 更新计数器
│   ├── self.episode_length_buf += 1
│   └── self._update_counters_each_step()
├── 🔍 预处理回调
│   └── self._pre_compute_observations_callback()
│       ├── 更新base_quat, rpy
│       ├── 计算base_lin_vel, base_ang_vel
│       └── 计算projected_gravity
├── 📋 任务更新
│   └── self._update_tasks_callback()
├── ❌ 检查终止条件
│   └── self._check_termination()
│       ├── self._update_reset_buf()
│       ├── self._update_timeout_buf()
│       └── self.reset_buf |= self.time_out_buf
├── 🎁 计算奖励
│   └── self._compute_reward()
│       └── for each reward_function:
│           ├── rew = reward_function() * scale
│           ├── 应用课程学习系数 [如果启用]
│           └── self.rew_buf += rew
├── 🔄 重置环境
│   ├── env_ids = self.reset_buf.nonzero().flatten()
│   └── self.reset_envs_idx(env_ids)
│       ├── self._reset_root_states(env_ids)
│       ├── self._reset_dofs(env_ids)
│       └── 其他重置操作
├── 🔄 刷新需要更新的环境
│   ├── refresh_env_ids = self.need_to_refresh_envs.nonzero().flatten()
│   ├── self.simulator.set_actor_root_state_tensor(refresh_env_ids, states)
│   └── self.simulator.set_dof_state_tensor(refresh_env_ids, dof_state)
├── 👁️ 计算观测
│   └── self._compute_observations()
│       ├── 初始化观测字典
│       ├── 设置噪声课程学习系数
│       ├── for obs_key, obs_config in config.obs.obs_dict.items():
│       │   └── parse_observation(self, obs_config, obs_buf, scales, noise)
│       └── 处理历史观测
├── 🔄 后处理回调
│   └── self._post_compute_observations_callback()
│       ├── self.last_actions = self.actions
│       ├── self.last_dof_pos = self.simulator.dof_pos
│       └── self.last_dof_vel = self.simulator.dof_vel
└── ✂️ 裁剪观测
    └── torch.clip(obs_val, -clip_obs, clip_obs)
```

## 📈 6. 策略更新阶段

### 6.1 训练步骤 (_training_step)
```
PPO._training_step()
├── loss_dict = self._init_loss_dict_at_training_step()
├── generator = storage.mini_batch_generator(num_mini_batches, num_epochs)
├── for policy_state_dict in generator:        # 遍历所有mini-batch
│   ├── 将数据移动到设备
│   └── loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
│       └── loss_dict = self._update_ppo(policy_state_dict, loss_dict)
├── 计算平均损失
└── self.storage.clear()                       # 清空存储器
```

### 6.2 PPO更新 (_update_ppo)
```
PPO._update_ppo(policy_state_dict, loss_dict)
├── 📥 提取batch数据
│   ├── actions_batch, advantages_batch, returns_batch
│   ├── old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
│   └── target_values_batch
├── 🎭 重新计算策略输出
│   ├── self._actor_act_step(policy_state_dict)
│   │   └── self.actor.act(obs_dict["actor_obs"])
│   ├── actions_log_prob_batch = self.actor.get_actions_log_prob(actions)
│   ├── value_batch = self._critic_eval_step(policy_state_dict)
│   └── entropy_batch = self.actor.entropy
├── 📊 计算PPO损失
│   ├── 🎯 Surrogate Loss
│   │   ├── ratio = exp(new_log_prob - old_log_prob)
│   │   ├── surrogate = -advantages * ratio
│   │   ├── surrogate_clipped = -advantages * clamp(ratio, 1-ε, 1+ε)
│   │   └── surrogate_loss = max(surrogate, surrogate_clipped).mean()
│   ├── 💰 Value Loss
│   │   └── value_loss = (returns - values)².mean()
│   └── 🎲 Entropy Loss
│       └── entropy_loss = entropy.mean()
├── 🔧 组合损失
│   ├── actor_loss = surrogate_loss - entropy_coef * entropy_loss
│   └── critic_loss = value_loss_coef * value_loss
├── 🔄 反向传播和优化
│   ├── self.actor_optimizer.zero_grad()
│   ├── self.critic_optimizer.zero_grad()
│   ├── actor_loss.backward()
│   ├── critic_loss.backward()
│   ├── 梯度裁剪
│   ├── self.actor_optimizer.step()
│   └── self.critic_optimizer.step()
└── 📊 更新损失字典
```

## 🔄 7. Motion Tracking特殊流程

### 7.1 Motion Tracking环境特殊处理
```
📁 humanoidverse/envs/motion_tracking/motion_tracking.py
LeggedRobotMotionTracking._pre_compute_observations_callback()
├── super()._pre_compute_observations_callback()  # 调用父类方法
├── 计算当前动作时间
│   └── motion_times = episode_length_buf * dt + motion_start_times
├── 从动作库获取参考状态
│   └── motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset)
└── 存储参考状态用于奖励计算
```

### 7.2 Motion Tracking重置流程
```
LeggedRobotMotionTracking._reset_root_states(env_ids)
├── 计算当前动作时间
├── 从动作库获取参考状态
├── 添加初始化噪声
│   ├── root_pos_noise, root_rot_noise
│   └── root_vel_noise, root_ang_vel_noise
└── 设置机器人状态为参考状态+噪声
```

这个完整的调用流程展示了HumanoidVerse框架从启动到一次训练迭代的所有关键步骤，帮助您理解代码的执行顺序和各组件之间的交互关系。
