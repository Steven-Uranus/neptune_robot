# Motion Tracking 详细函数调用顺序

本文档详细列出了 `humanoidverse/envs/motion_tracking/motion_tracking.py` 文件中Motion Tracking任务的具体函数调用顺序。

## 🎯 1. LeggedRobotMotionTracking.__init__() - 初始化

### 1.1 函数调用顺序
```python
LeggedRobotMotionTracking.__init__(config, device)
├── super().__init__(config, device)        # 调用LeggedRobotBase初始化
├── self._init_motion_lib()                 # 初始化动作库
│   ├── self.config.robot.motion.step_dt = self.dt
│   ├── self._motion_lib = MotionLibRobot(config, num_envs, device)
│   ├── if self.is_evaluating:
│   │   └── self._motion_lib.load_motions(random_sample=False)
│   ├── else:
│   │   └── self._motion_lib.load_motions(random_sample=True)
│   ├── res = self._resample_motion_times(torch.arange(self.num_envs))
│   ├── self.motion_dt = self._motion_lib._motion_dt
│   ├── self.motion_start_idx = 0
│   └── self.num_motions = self._motion_lib._num_unique_motions
├── self._init_motion_extend()              # 初始化动作扩展
├── self._init_tracking_config()            # 初始化追踪配置
└── self._init_save_motion()                # 初始化动作保存
```

## 🔄 2. Motion Tracking特殊的step流程

### 2.1 _pre_compute_observations_callback() - 观测前回调
```python
LeggedRobotMotionTracking._pre_compute_observations_callback()
├── super()._pre_compute_observations_callback()  # 调用父类方法
├── 1. 计算当前动作时间
│   ├── offset = self.env_origins
│   ├── B = self.motion_ids.shape[0]
│   └── motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
├── 2. 从动作库获取参考状态
│   └── motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset)
│       ├── 获取参考根位置: motion_res['root_pos']
│       ├── 获取参考根旋转: motion_res['root_rot']
│       ├── 获取参考根速度: motion_res['root_vel']
│       ├── 获取参考关节位置: motion_res['dof_pos']
│       ├── 获取参考关节速度: motion_res['dof_vel']
│       └── 获取参考身体位置: motion_res['body_pos'] [如果有]
└── 3. 存储参考状态用于奖励计算
    ├── self.ref_root_pos = motion_res['root_pos']
    ├── self.ref_root_rot = motion_res['root_rot']
    ├── self.ref_root_vel = motion_res['root_vel']
    ├── self.ref_dof_pos = motion_res['dof_pos']
    ├── self.ref_dof_vel = motion_res['dof_vel']
    └── self.ref_body_pos = motion_res['body_pos'] [如果有]
```

### 2.2 _update_timeout_buf() - 更新超时缓冲区
```python
LeggedRobotMotionTracking._update_timeout_buf()
├── super()._update_timeout_buf()           # 调用父类方法
└── if self.config.termination.terminate_when_motion_end:
    ├── current_time = episode_length_buf * dt + motion_start_times
    └── self.time_out_buf |= current_time > self.motion_len
```

## 🔄 3. 环境重置相关函数

### 3.1 _reset_root_states() - 重置根状态
```python
LeggedRobotMotionTracking._reset_root_states(env_ids)
├── 1. 计算当前动作时间
│   ├── motion_times = episode_length_buf * dt + motion_start_times
│   └── offset = self.env_origins
├── 2. 从动作库获取参考状态
│   └── motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset)
├── 3. 处理自定义原点情况 [如果启用]
│   ├── if self.custom_origins:
│   │   └── self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
├── 4. 添加初始化噪声
│   ├── root_pos_noise = config.init_noise_scale.root_pos * noise_level
│   ├── root_rot_noise = config.init_noise_scale.root_rot * noise_level
│   ├── root_vel_noise = config.init_noise_scale.root_vel * noise_level
│   └── root_ang_vel_noise = config.init_noise_scale.root_ang_vel * noise_level
├── 5. 设置根位置和旋转
│   ├── root_pos = motion_res['root_pos'][env_ids]
│   ├── root_rot = motion_res['root_rot'][env_ids]
│   ├── self.simulator.robot_root_states[env_ids, :3] = root_pos + noise
│   └── 根据仿真器类型设置旋转
│       ├── if simulator == 'isaacgym':
│       │   └── robot_root_states[env_ids, 3:7] = quat_mul(random_quat, root_rot)
│       ├── elif simulator == 'isaacsim':
│       │   └── robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(...))
│       └── elif simulator == 'genesis':
│           └── robot_root_states[env_ids, 3:7] = quat_mul(random_quat, root_rot)
├── 6. 设置根速度
│   ├── root_vel = motion_res['root_vel'][env_ids]
│   ├── self.simulator.robot_root_states[env_ids, 7:10] = root_vel[:, :3] + noise
│   └── self.simulator.robot_root_states[env_ids, 10:13] = root_vel[:, 3:6] + noise
└── 7. 处理地形高度调整 [如果需要]
    └── self.simulator.robot_root_states[env_ids, 2] += terrain_height_adjustment
```

### 3.2 _reset_dofs() - 重置关节状态
```python
LeggedRobotMotionTracking._reset_dofs(env_ids)
├── 1. 计算当前动作时间
│   ├── motion_times = episode_length_buf * dt + motion_start_times
│   └── offset = self.env_origins
├── 2. 从动作库获取参考关节状态
│   └── motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset)
├── 3. 添加关节噪声
│   ├── dof_pos_noise = config.init_noise_scale.dof_pos * noise_level
│   ├── dof_vel_noise = config.init_noise_scale.dof_vel * noise_level
│   ├── dof_pos = motion_res['dof_pos'][env_ids]
│   └── dof_vel = motion_res['dof_vel'][env_ids]
├── 4. 设置关节状态
│   ├── self.simulator.dof_pos[env_ids] = dof_pos + noise
│   └── self.simulator.dof_vel[env_ids] = dof_vel + noise
```

### 3.3 _resample_motion_times() - 重新采样动作时间
```python
LeggedRobotMotionTracking._resample_motion_times(env_ids)
├── 1. 获取动作信息
│   ├── motion_ids = self._motion_lib.sample_motions(len(env_ids))
│   ├── motion_len = self._motion_lib.get_motion_length(motion_ids)
│   └── self.motion_ids[env_ids] = motion_ids
├── 2. 采样动作开始时间
│   ├── if self.config.motion.sample_time_min >= 0:
│   │   ├── sample_time_min = config.motion.sample_time_min
│   │   ├── sample_time_max = motion_len - config.motion.sample_time_max
│   │   └── motion_start_times = uniform_sample(sample_time_min, sample_time_max)
│   └── else:
│       └── motion_start_times = torch.zeros_like(motion_len)
├── 3. 存储动作信息
│   ├── self.motion_start_times[env_ids] = motion_start_times
│   ├── self.motion_len[env_ids] = motion_len
│   └── return motion_res
```

## 🎁 4. Motion Tracking特殊奖励函数

### 4.1 典型的Motion Tracking奖励函数调用
```python
# 根位置追踪奖励
def _reward_tracking_root_pos(self):
├── pos_err = torch.norm(self.simulator.robot_root_states[:, :3] - self.ref_root_pos, dim=-1)
└── return torch.exp(-pos_err / sigma)

# 根旋转追踪奖励  
def _reward_tracking_root_rot(self):
├── rot_err = quat_diff_rad(self.simulator.robot_root_states[:, 3:7], self.ref_root_rot)
└── return torch.exp(-rot_err / sigma)

# 关节位置追踪奖励
def _reward_tracking_dof_pos(self):
├── dof_err = torch.norm(self.simulator.dof_pos - self.ref_dof_pos, dim=-1)
└── return torch.exp(-dof_err / sigma)

# 关节速度追踪奖励
def _reward_tracking_dof_vel(self):
├── vel_err = torch.norm(self.simulator.dof_vel - self.ref_dof_vel, dim=-1)
└── return torch.exp(-vel_err / sigma)

# 身体位置追踪奖励 [如果有]
def _reward_tracking_body_pos(self):
├── body_pos_err = torch.norm(self.body_pos - self.ref_body_pos, dim=-1)
└── return torch.exp(-body_pos_err / sigma)
```

## 🔧 5. 任务管理函数

### 5.1 next_task() - 下一个任务 [评估模式]
```python
LeggedRobotMotionTracking.next_task()
├── self.motion_start_idx += self.num_envs
├── if self.motion_start_idx >= self.num_motions:
│   └── self.motion_start_idx = 0
├── self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
└── self.reset_all()
```

### 5.2 resample_motion() - 重新采样动作
```python
LeggedRobotMotionTracking.resample_motion()
├── self._motion_lib.load_motions(random_sample=True)
└── self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
```

## 📊 6. 观测计算相关

### 6.1 Motion Tracking特殊观测项
```python
# 参考根位置观测
def _get_obs_ref_root_pos(self):
└── return self.ref_root_pos

# 参考根旋转观测  
def _get_obs_ref_root_rot(self):
└── return self.ref_root_rot

# 参考关节位置观测
def _get_obs_ref_dof_pos(self):
└── return self.ref_dof_pos - self.default_dof_pos

# 参考关节速度观测
def _get_obs_ref_dof_vel(self):
└── return self.ref_dof_vel

# 动作相位观测 [如果启用]
def _get_obs_motion_phase(self):
├── current_time = self.episode_length_buf * self.dt + self.motion_start_times
├── motion_phase = current_time / self.motion_len
└── return motion_phase.unsqueeze(-1)
```

## 🔄 7. 完整的Motion Tracking Step流程

### 7.1 集成的step调用顺序
```python
LeggedRobotMotionTracking.step(actor_state)
├── 继承自LeggedRobotBase.step()
├── _pre_physics_step(actions)              # 预处理
├── _physics_step()                         # 物理仿真
└── _post_physics_step()                    # 后处理
    ├── _refresh_sim_tensors()
    ├── _update_counters_each_step()
    ├── _pre_compute_observations_callback() # 🎯 Motion Tracking特殊处理
    │   ├── 计算当前动作时间
    │   ├── 从动作库获取参考状态
    │   └── 存储参考状态
    ├── _update_tasks_callback()
    ├── _check_termination()
    │   └── _update_timeout_buf()           # 🎯 检查动作结束
    ├── _compute_reward()                   # 🎯 使用参考状态计算追踪奖励
    ├── reset_envs_idx()                    # 🎯 使用参考状态重置
    │   ├── _reset_root_states()
    │   └── _reset_dofs()
    ├── _compute_observations()             # 🎯 包含参考状态观测
    └── _post_compute_observations_callback()
```

这个详细的Motion Tracking函数调用顺序展示了动作追踪任务如何在基础环境框架上添加特殊的动作库管理、参考状态计算和追踪奖励功能。
