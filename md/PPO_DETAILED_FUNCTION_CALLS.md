# PPO.py 详细函数调用顺序

本文档详细列出了 `humanoidverse/agents/ppo/ppo.py` 文件中每个函数的具体调用顺序和内部逻辑。

## 🎯 1. PPO.learn() - 主训练循环

### 1.1 函数调用顺序
```python
PPO.learn()
├── 1. 初始化设置
│   ├── 设置随机episode长度 [可选]
│   ├── self.env.reset_all()                    # 重置所有环境
│   ├── obs_dict[obs_key].to(self.device)       # 转移观测到设备
│   └── self._train_mode()                      # 设置训练模式
│
├── 2. 主训练循环 for it in range(current_iter, tot_iter):
│   ├── time.time()                             # 记录开始时间
│   ├── obs_dict = self._rollout_step(obs_dict) # 🔄 数据收集阶段
│   ├── loss_dict = self._training_step()       # 📈 策略更新阶段
│   ├── time.time()                             # 记录结束时间
│   ├── self._post_epoch_logging(log_dict)      # 📊 日志记录
│   ├── self.save() [按间隔]                    # 💾 保存模型
│   └── self.ep_infos.clear()                   # 清空episode信息
│
└── 3. 训练结束
    ├── self.current_learning_iteration += num_iterations
    └── self.save()                             # 最终保存
```

## 🔄 2. PPO._rollout_step() - 数据收集阶段

### 2.1 函数调用顺序
```python
PPO._rollout_step(obs_dict)
├── with torch.inference_mode():                # 推理模式，不计算梯度
└── for i in range(self.num_steps_per_env):     # 每个环境收集N步
    ├── 🎭 策略推理阶段
    │   ├── policy_state_dict = {}
    │   ├── policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
    │   │   ├── actions = self._actor_act_step(obs_dict)
    │   │   │   └── return self.actor.act(obs_dict["actor_obs"])
    │   │   │       ├── self.actor.update_distribution(actor_obs)
    │   │   │       │   ├── mean = self.actor_module(actor_obs)
    │   │   │       │   └── self.distribution = Normal(mean, std)
    │   │   │       └── return self.distribution.sample()
    │   │   ├── policy_state_dict["actions"] = actions
    │   │   ├── action_mean = self.actor.action_mean.detach()
    │   │   ├── action_sigma = self.actor.action_std.detach()
    │   │   ├── actions_log_prob = self.actor.get_actions_log_prob(actions).detach()
    │   │   ├── policy_state_dict["action_mean"] = action_mean
    │   │   ├── policy_state_dict["action_sigma"] = action_sigma
    │   │   ├── policy_state_dict["actions_log_prob"] = actions_log_prob
    │   │   └── return policy_state_dict
    │   ├── values = self._critic_eval_step(obs_dict).detach()
    │   │   └── return self.critic.evaluate(obs_dict["critic_obs"])
    │   └── policy_state_dict["values"] = values
    │
    ├── 💾 存储状态到缓冲区
    │   ├── for obs_key in obs_dict.keys():
    │   │   └── self.storage.update_key(obs_key, obs_dict[obs_key])
    │   └── for obs_ in policy_state_dict.keys():
    │       └── self.storage.update_key(obs_, policy_state_dict[obs_])
    │
    ├── 🌍 环境交互阶段
    │   ├── actions = policy_state_dict["actions"]
    │   ├── actor_state = {"actions": actions}
    │   ├── obs_dict, rewards, dones, infos = self.env.step(actor_state)
    │   ├── obs_dict[obs_key].to(self.device)    # 转移观测到设备
    │   └── rewards.to(self.device), dones.to(self.device)
    │
    ├── 📊 记录和处理
    │   ├── self.episode_env_tensors.add(infos["to_log"])
    │   ├── rewards_stored = rewards.clone().unsqueeze(1)
    │   ├── 处理timeout奖励 [如果有time_outs]
    │   │   └── rewards_stored += γ * values * time_outs
    │   ├── self.storage.update_key('rewards', rewards_stored)
    │   ├── self.storage.update_key('dones', dones.unsqueeze(1))
    │   ├── self.storage.increment_step()
    │   └── self._process_env_step(rewards, dones, infos)
    │       ├── self.actor.reset(dones)
    │       └── self.critic.reset(dones)
    │
    └── 📈 统计信息更新
        ├── if 'episode' in infos:
        │   └── self.ep_infos.append(infos['episode'])
        ├── self.cur_reward_sum += rewards
        ├── self.cur_episode_length += 1
        ├── new_ids = (dones > 0).nonzero(as_tuple=False)
        ├── self.rewbuffer.extend(self.cur_reward_sum[new_ids])
        ├── self.lenbuffer.extend(self.cur_episode_length[new_ids])
        ├── self.cur_reward_sum[new_ids] = 0
        └── self.cur_episode_length[new_ids] = 0
│
└── 🧮 计算GAE和回报
    ├── self.stop_time = time.time()
    ├── self.collection_time = stop_time - start_time
    ├── returns, advantages = self._compute_returns(obs_dict, policy_state_dict)
    ├── self.storage.batch_update_data('returns', returns)
    ├── self.storage.batch_update_data('advantages', advantages)
    └── return obs_dict
```

## 🧮 3. PPO._compute_returns() - GAE计算

### 3.1 函数调用顺序
```python
PPO._compute_returns(last_obs_dict, policy_state_dict)
├── 1. 初始化
│   ├── last_values = self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
│   ├── advantage = 0
│   ├── values = policy_state_dict['values']
│   ├── dones = policy_state_dict['dones']
│   ├── rewards = policy_state_dict['rewards']
│   ├── 转移所有张量到设备
│   ├── returns = torch.zeros_like(values)
│   └── num_steps = returns.shape[0]
│
├── 2. 反向计算GAE for step in reversed(range(num_steps)):
│   ├── if step == num_steps - 1:
│   │   └── next_values = last_values        # 最后一步使用bootstrap
│   ├── else:
│   │   └── next_values = values[step + 1]   # 其他步使用下一步价值
│   ├── next_is_not_terminal = 1.0 - dones[step].float()
│   ├── delta = rewards[step] + γ*next_values - values[step]  # TD误差
│   ├── advantage = delta + γ*λ*advantage    # GAE优势函数
│   └── returns[step] = advantage + values[step]  # 回报
│
└── 3. 标准化优势函数
    ├── advantages = returns - values
    ├── advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    └── return returns, advantages
```

## 📈 4. PPO._training_step() - 策略更新阶段

### 4.1 函数调用顺序
```python
PPO._training_step()
├── 1. 初始化
│   ├── loss_dict = self._init_loss_dict_at_training_step()
│   │   ├── loss_dict['Value'] = 0
│   │   ├── loss_dict['Surrogate'] = 0
│   │   └── loss_dict['Entropy'] = 0
│   └── generator = self.storage.mini_batch_generator(num_mini_batches, num_epochs)
│
├── 2. Mini-batch训练循环 for policy_state_dict in generator:
│   ├── for policy_state_key in policy_state_dict.keys():
│   │   └── policy_state_dict[policy_state_key].to(self.device)
│   └── loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
│       └── loss_dict = self._update_ppo(policy_state_dict, loss_dict)
│
├── 3. 计算平均损失
│   ├── num_updates = num_learning_epochs * num_mini_batches
│   └── for key in loss_dict.keys():
│       └── loss_dict[key] /= num_updates
│
└── 4. 清理
    ├── self.storage.clear()
    └── return loss_dict
```

## 🔧 5. PPO._update_ppo() - PPO核心更新

### 5.1 函数调用顺序
```python
PPO._update_ppo(policy_state_dict, loss_dict)
├── 1. 提取batch数据
│   ├── actions_batch = policy_state_dict['actions']
│   ├── target_values_batch = policy_state_dict['values']
│   ├── advantages_batch = policy_state_dict['advantages']
│   ├── returns_batch = policy_state_dict['returns']
│   ├── old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
│   ├── old_mu_batch = policy_state_dict['action_mean']
│   └── old_sigma_batch = policy_state_dict['action_sigma']
│
├── 2. 重新计算策略输出
│   ├── self._actor_act_step(policy_state_dict)
│   │   └── return self.actor.act(obs_dict["actor_obs"])
│   ├── actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
│   ├── value_batch = self._critic_eval_step(policy_state_dict)
│   ├── mu_batch = self.actor.action_mean
│   ├── sigma_batch = self.actor.action_std
│   └── entropy_batch = self.actor.entropy
│
├── 3. 自适应学习率调整 [如果启用]
│   ├── with torch.inference_mode():
│   ├── kl = torch.sum(torch.log(σ_new/σ_old) + (σ_old² + (μ_old-μ_new)²)/(2σ_new²) - 0.5)
│   ├── kl_mean = torch.mean(kl)
│   ├── if kl_mean > desired_kl * 2.0:
│   │   └── learning_rate /= 1.5
│   ├── elif kl_mean < desired_kl / 2.0:
│   │   └── learning_rate *= 1.5
│   └── 更新优化器学习率
│
├── 4. 计算PPO损失
│   ├── 🎯 Surrogate Loss
│   │   ├── ratio = exp(new_log_prob - old_log_prob)
│   │   ├── surrogate = -advantages * ratio
│   │   ├── surrogate_clipped = -advantages * clamp(ratio, 1-ε, 1+ε)
│   │   └── surrogate_loss = max(surrogate, surrogate_clipped).mean()
│   ├── 💰 Value Loss
│   │   ├── if use_clipped_value_loss:
│   │   │   ├── value_clipped = old_values + clamp(new_values - old_values, -ε, ε)
│   │   │   ├── value_losses = (new_values - returns)²
│   │   │   ├── value_losses_clipped = (value_clipped - returns)²
│   │   │   └── value_loss = max(value_losses, value_losses_clipped).mean()
│   │   └── else: value_loss = (returns - new_values)².mean()
│   ├── 🎲 Entropy Loss
│   │   └── entropy_loss = entropy_batch.mean()
│   ├── actor_loss = surrogate_loss - entropy_coef * entropy_loss
│   └── critic_loss = value_loss_coef * value_loss
│
├── 5. 反向传播和优化
│   ├── self.actor_optimizer.zero_grad()
│   ├── self.critic_optimizer.zero_grad()
│   ├── actor_loss.backward()
│   ├── critic_loss.backward()
│   ├── nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
│   ├── nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
│   ├── self.actor_optimizer.step()
│   └── self.critic_optimizer.step()
│
└── 6. 更新损失字典
    ├── loss_dict['Value'] += value_loss.item()
    ├── loss_dict['Surrogate'] += surrogate_loss.item()
    ├── loss_dict['Entropy'] += entropy_loss.item()
    └── return loss_dict
```

## 🔧 6. 辅助函数调用

### 6.1 其他重要函数
```python
# Actor相关
PPO._actor_act_step(obs_dict)
└── return self.actor.act(obs_dict["actor_obs"])

# Critic相关  
PPO._critic_eval_step(obs_dict)
└── return self.critic.evaluate(obs_dict["critic_obs"])

# 环境步骤处理
PPO._process_env_step(rewards, dones, infos)
├── self.actor.reset(dones)
└── self.critic.reset(dones)

# 训练模式设置
PPO._train_mode()
├── self.actor.train()
└── self.critic.train()

# 日志记录
PPO._post_epoch_logging(log_dict)
├── 计算FPS和统计信息
├── self.episode_env_tensors.mean_and_clear()
├── self._logging_to_writer(log_dict, train_log_dict, env_log_dict)
└── 打印训练信息
```

这个详细的函数调用顺序展示了PPO算法的完整执行流程，从数据收集到策略更新的每一个步骤都有清晰的调用链。

---

# Environment.step() 详细函数调用顺序

## 🌍 1. LeggedRobotBase.step() - 环境主步骤

### 1.1 函数调用顺序
```python
LeggedRobotBase.step(actor_state)
├── actions = actor_state["actions"]
├── self._pre_physics_step(actions)         # 🔧 预处理阶段
├── self._physics_step()                    # ⚡ 物理仿真阶段
├── self._post_physics_step()               # 🔄 后处理阶段
└── return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras
```

## 🔧 2. _pre_physics_step() - 预处理阶段

### 2.1 函数调用顺序
```python
LeggedRobotBase._pre_physics_step(actions)
├── 1. 动作裁剪
│   ├── clip_action_limit = self.config.robot.control.action_clip_value
│   └── self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit)
├── 2. 记录裁剪统计
│   └── self.log_dict["action_clip_frac"] = (actions.abs() == clip_limit).sum() / actions.numel()
└── 3. 控制延迟处理 [如果启用域随机化]
    ├── if self.config.domain_rand.randomize_ctrl_delay:
    ├── self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()  # 队列后移
    ├── self.action_queue[:, 0] = self.actions.clone()               # 新动作入队
    ├── self.actions_after_delay = self.action_queue[env_ids, delay_idx].clone()
    └── else: self.actions_after_delay = self.actions.clone()
```

## ⚡ 3. _physics_step() - 物理仿真阶段

### 3.1 函数调用顺序
```python
LeggedRobotBase._physics_step()
├── self.render()                           # 渲染 [如果启用]
└── for _ in range(control_decimation):     # 控制抽取循环
    ├── self._apply_force_in_physics_step()
    │   ├── self.torques = self._compute_torques(self.actions_after_delay)
    │   │   ├── actions_scaled = actions * action_scale
    │   │   ├── if control_type == "P":     # 位置控制
    │   │   │   └── torques = kp*(target_pos - current_pos) - kd*current_vel
    │   │   ├── elif control_type == "V":   # 速度控制
    │   │   │   └── torques = kp*(target_vel - current_vel) - kd*vel_diff
    │   │   ├── elif control_type == "T":   # 扭矩控制
    │   │   │   └── torques = actions_scaled
    │   │   ├── 添加随机力干扰 [如果启用]
    │   │   └── torch.clip(torques, -torque_limits, torque_limits)
    │   └── self.simulator.apply_torques_at_dof(self.torques)
    └── self.simulator.simulate_at_each_physics_step()
```

## 🔄 4. _post_physics_step() - 后处理阶段

### 4.1 函数调用顺序
```python
LeggedRobotBase._post_physics_step()
├── 1. 刷新仿真状态
│   └── self._refresh_sim_tensors()
│       └── self.simulator.refresh_sim_tensors()
├── 2. 更新计数器
│   ├── self.episode_length_buf += 1
│   ├── self._update_counters_each_step()
│   └── self.last_episode_length_buf = self.episode_length_buf.clone()
├── 3. 预处理回调
│   ├── self._pre_compute_observations_callback()
│   │   ├── self.base_quat[:] = self.simulator.base_quat[:]
│   │   ├── self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat)
│   │   ├── self.base_lin_vel[:] = quat_rotate_inverse(base_quat, root_states[:, 7:10])
│   │   ├── self.base_ang_vel[:] = quat_rotate_inverse(base_quat, root_states[:, 10:13])
│   │   └── self.projected_gravity[:] = quat_rotate_inverse(base_quat, gravity_vec)
│   └── self._update_tasks_callback()        # 任务特定更新
├── 4. 检查终止和计算奖励
│   ├── self._check_termination()
│   │   ├── self.reset_buf[:] = 0
│   │   ├── self.time_out_buf[:] = 0
│   │   ├── self._update_reset_buf()         # 更新重置缓冲区
│   │   ├── self._update_timeout_buf()       # 更新超时缓冲区
│   │   └── self.reset_buf |= self.time_out_buf
│   └── self._compute_reward()               # 计算奖励 (见5.奖励计算)
├── 5. 重置环境
│   ├── env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
│   └── self.reset_envs_idx(env_ids)         # 重置指定环境 (见6.环境重置)
├── 6. 刷新需要更新的环境
│   ├── refresh_env_ids = self.need_to_refresh_envs.nonzero().flatten()
│   ├── if len(refresh_env_ids) > 0:
│   │   ├── self.simulator.set_actor_root_state_tensor(refresh_env_ids, all_root_states)
│   │   ├── self.simulator.set_dof_state_tensor(refresh_env_ids, dof_state)
│   │   └── self.need_to_refresh_envs[refresh_env_ids] = False
├── 7. 计算观测
│   └── self._compute_observations()         # 计算观测 (见7.观测计算)
├── 8. 后处理回调
│   └── self._post_compute_observations_callback()
│       ├── self.last_actions[:] = self.actions[:]
│       ├── self.last_dof_pos[:] = self.simulator.dof_pos[:]
│       ├── self.last_dof_vel[:] = self.simulator.dof_vel[:]
│       └── self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]
└── 9. 裁剪观测
    ├── clip_obs = self.config.normalization.clip_observations
    └── for obs_key, obs_val in self.obs_buf_dict.items():
        └── self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)
```

## 🎁 5. _compute_reward() - 奖励计算

### 5.1 函数调用顺序
```python
LeggedRobotBase._compute_reward()
├── self.rew_buf[:] = 0.                    # 重置奖励缓冲区
├── for reward_name, reward_func in self.reward_functions.items():
│   ├── rew = reward_func() * reward_scale  # 计算单项奖励
│   ├── 应用课程学习系数 [如果启用]
│   │   └── rew *= curriculum_factor
│   ├── self.rew_buf += rew                 # 累加到总奖励
│   └── self.episode_sums[reward_name] += rew  # 记录episode累计奖励
└── 应用奖励课程学习 [如果启用]
    └── self.rew_buf *= self.current_reward_curriculum_value
```

## 🔄 6. reset_envs_idx() - 环境重置

### 6.1 函数调用顺序
```python
LeggedRobotBase.reset_envs_idx(env_ids)
├── if len(env_ids) == 0: return           # 无需重置则返回
├── self._reset_dofs(env_ids)              # 重置关节状态
├── self._reset_root_states(env_ids)       # 重置根状态
├── self._resample_commands(env_ids)       # 重新采样命令
├── 重置各种缓冲区
│   ├── self.reset_buf[env_ids] = 1
│   ├── self.episode_length_buf[env_ids] = 0
│   ├── self.episode_sums[reward_name][env_ids] = 0
│   └── 其他任务特定缓冲区重置
├── self.extras["episode"] = {}            # 记录episode信息
└── self.need_to_refresh_envs[env_ids] = True  # 标记需要刷新
```

## 👁️ 7. _compute_observations() - 观测计算

### 7.1 函数调用顺序
```python
LeggedRobotBase._compute_observations()
├── 1. 初始化观测字典
│   ├── self.obs_buf_dict_raw = {}          # 原始观测
│   └── self.hist_obs_dict = {}             # 历史观测
├── 2. 设置噪声课程学习
│   ├── if self.add_noise_currculum:
│   │   └── noise_extra_scale = self.current_noise_curriculum_value
│   └── else: noise_extra_scale = 1.
├── 3. 计算各种观测项
│   └── for obs_key, obs_config in self.config.obs.obs_dict.items():
│       ├── self.obs_buf_dict_raw[obs_key] = dict()
│       └── parse_observation(
│           self,                           # 环境实例
│           obs_config,                     # 观测配置
│           self.obs_buf_dict_raw[obs_key], # 输出字典
│           obs_scales,                     # 观测缩放
│           noise_scales,                   # 噪声缩放
│           noise_extra_scale               # 额外噪声
│       )
├── 4. 计算历史观测
│   ├── history_obs_list = self.history_handler.history.keys()
│   └── parse_observation(self, history_obs_list, self.hist_obs_dict, ...)
├── 5. 后处理观测
│   └── self._post_config_observation_callback()
│       ├── self.obs_buf_dict = dict()
│       └── for obs_key, obs_config in self.config.obs.obs_dict.items():
│           ├── obs_keys = sorted(obs_config)
│           └── self.obs_buf_dict[obs_key] = torch.cat([raw_obs[key] for key in obs_keys])
```

这个详细的环境step函数调用顺序展示了从动作输入到观测输出的完整流程，包括物理仿真、奖励计算、环境重置等所有关键步骤。
