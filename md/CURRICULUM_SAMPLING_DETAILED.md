# HumanoidVerse 课程式采样详细分析

本文档详细列出了HumanoidVerse项目中所有的课程式采样(Curriculum Sampling)和课程学习(Curriculum Learning)机制。

## 🎯 1. 课程学习概述

HumanoidVerse框架实现了多种课程学习机制，通过动态调整训练难度来提高学习效率和稳定性：

### 1.1 课程学习类型
- **奖励课程学习** (Reward Curriculum)
- **观测噪声课程学习** (Observation Noise Curriculum)
- **奖励限制课程学习** (Reward Limits Curriculum)
- **终止条件课程学习** (Termination Curriculum)
- **地形课程学习** (Terrain Curriculum)
- ~~**动作采样课程学习** (Motion Sampling Curriculum)~~ ❌ **不是课程式**

## 🏆 2. 奖励课程学习 (Reward Penalty Curriculum)

### 2.1 实现位置
```
📁 humanoidverse/envs/legged_base_task/legged_robot_base.py
├── _update_reward_penalty_curriculum()     # 更新奖励惩罚课程
├── _compute_reward()                       # 应用课程学习系数
└── _reset_tasks_callback()                 # 在重置时更新课程
```

### 2.2 核心实现
```python
def _update_reward_penalty_curriculum(self):
    """
    基于平均episode长度更新奖励惩罚课程
    
    逻辑：
    - episode长度短 → 增加惩罚 (降低惩罚系数)
    - episode长度长 → 减少惩罚 (提高惩罚系数)
    """
    if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
        # episode太短，增加惩罚强度
        self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
    elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
        # episode足够长，减少惩罚强度
        self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)
    
    # 限制在指定范围内
    self.reward_penalty_scale = np.clip(
        self.reward_penalty_scale,
        self.config.rewards.reward_min_penalty_scale,
        self.config.rewards.reward_max_penalty_scale
    )
```

### 2.3 配置参数
```yaml
# humanoidverse/config/rewards/loco/reward_h1_locomotion_10dof.yaml
reward_penalty_curriculum: False
reward_initial_penalty_scale: 1.0
reward_min_penalty_scale: 0.0
reward_max_penalty_scale: 1.0
reward_penalty_level_down_threshold: 400
reward_penalty_level_up_threshold: 700
reward_penalty_degree: 0.00001

# 受影响的奖励项
reward_penalty_reward_names: [
  "penalty_torques",
  "penalty_dof_acc", 
  "penalty_dof_vel",
  "penalty_action_rate",
  "penalty_feet_contact_forces",
  "limits_dof_pos",
  "limits_dof_vel"
]
```

## 🔊 3. 观测噪声课程学习 (Observation Noise Curriculum)

### 3.1 实现位置
```
📁 humanoidverse/envs/legged_base_task/legged_robot_base.py
├── _update_obs_noise_curriculum()          # 更新观测噪声课程
├── _compute_observations()                 # 应用噪声课程系数
└── _reset_tasks_callback()                 # 在重置时更新课程
```

### 3.2 核心实现
```python
def _update_obs_noise_curriculum(self):
    """
    基于平均episode长度动态调整观测噪声水平
    
    逻辑：
    - episode长度短 → 减少噪声 (降低难度)
    - episode长度长 → 增加噪声 (提高难度)
    """
    if self.average_episode_length < self.config.obs.soft_dof_pos_curriculum_level_down_threshold:
        # episode太短，减少噪声
        self.current_noise_curriculum_value *= (1 - self.config.obs.soft_dof_pos_curriculum_degree)
    elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
        # episode足够长，增加噪声
        self.current_noise_curriculum_value *= (1 + self.config.obs.soft_dof_pos_curriculum_degree)
    
    # 限制噪声范围
    self.current_noise_curriculum_value = np.clip(
        self.current_noise_curriculum_value,
        self.config.obs.noise_value_min,
        self.config.obs.noise_value_max
    )

def _compute_observations(self):
    # 应用噪声课程学习系数
    if self.add_noise_currculum:
        noise_extra_scale = self.current_noise_curriculum_value
    else:
        noise_extra_scale = 1.0
    
    # 在观测计算中使用噪声系数
    parse_observation(
        self, obs_config, obs_buf,
        obs_scales, noise_scales, 
        noise_extra_scale  # 课程学习的额外噪声系数
    )
```

### 3.3 配置参数
```yaml
# humanoidverse/config/obs/loco/leggedloco_obs_singlestep_withlinvel.yaml
add_noise_currculum: False
noise_initial_value: 0.05
noise_value_max: 1.00
noise_value_min: 0.00001
soft_dof_pos_curriculum_degree: 0.00001
soft_dof_pos_curriculum_level_down_threshold: 100
soft_dof_pos_curriculum_level_up_threshold: 900
```

## ⚖️ 4. 奖励限制课程学习 (Reward Limits Curriculum)

### 4.1 实现位置
```
📁 humanoidverse/envs/legged_base_task/legged_robot_base.py
├── _update_reward_limits_curriculum()      # 更新奖励限制课程
├── 各种limits奖励函数                      # 应用课程学习的限制值
└── _reset_tasks_callback()                 # 在重置时更新课程
```

### 4.2 核心实现
```python
def _update_reward_limits_curriculum(self):
    """
    动态调整关节位置、速度、扭矩的软限制
    """
    # 关节位置限制课程
    if self.use_reward_limits_dof_pos_curriculum:
        if self.average_episode_length < threshold_down:
            # episode短，放宽限制
            self.soft_dof_pos_curriculum_value *= (1 + degree)
        elif self.average_episode_length > threshold_up:
            # episode长，收紧限制
            self.soft_dof_pos_curriculum_value *= (1 - degree)
        
        self.soft_dof_pos_curriculum_value = np.clip(
            self.soft_dof_pos_curriculum_value,
            min_limit, max_limit
        )
    
    # 关节速度限制课程 (类似逻辑)
    if self.use_reward_limits_dof_vel_curriculum:
        # ... 类似的更新逻辑
    
    # 扭矩限制课程 (类似逻辑)  
    if self.use_reward_limits_torque_curriculum:
        # ... 类似的更新逻辑
```

### 4.3 配置参数
```yaml
# humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml
reward_limits_curriculum:
  soft_dof_pos_curriculum: True
  soft_dof_pos_initial_limit: 1.15
  soft_dof_pos_max_limit: 1.15
  soft_dof_pos_min_limit: 0.95
  soft_dof_pos_curriculum_degree: 0.00000025
  soft_dof_pos_curriculum_level_down_threshold: 40
  soft_dof_pos_curriculum_level_up_threshold: 42

  soft_dof_vel_curriculum: True
  soft_dof_vel_initial_limit: 1.15
  soft_dof_vel_max_limit: 1.25
  soft_dof_vel_min_limit: 0.95
  soft_dof_vel_curriculum_degree: 0.00000025
  soft_dof_vel_curriculum_level_down_threshold: 40
  soft_dof_vel_curriculum_level_up_threshold: 42
```

## ❌ 5. 终止条件课程学习 (Termination Curriculum)

### 5.1 实现位置 (Motion Tracking)
```
📁 humanoidverse/envs/motion_tracking/motion_tracking.py
├── _update_terminate_when_motion_far_curriculum()  # 更新终止距离课程
├── _reset_tasks_callback()                         # 在重置时更新课程
└── _check_termination()                            # 应用动态终止阈值
```

### 5.2 核心实现
```python
def _update_terminate_when_motion_far_curriculum(self):
    """
    动态调整动作偏离终止阈值
    
    逻辑：
    - episode短 → 放宽终止条件 (增加阈值)
    - episode长 → 收紧终止条件 (减少阈值)
    """
    if self.average_episode_length < threshold_down:
        # episode太短，放宽终止条件
        self.terminate_when_motion_far_threshold *= (1 + degree)
    elif self.average_episode_length > threshold_up:
        # episode足够长，收紧终止条件
        self.terminate_when_motion_far_threshold *= (1 - degree)
    
    # 限制阈值范围
    self.terminate_when_motion_far_threshold = np.clip(
        self.terminate_when_motion_far_threshold,
        threshold_min, threshold_max
    )
```

## 🏔️ 6. 地形课程学习 (Terrain Curriculum)

### 6.1 实现位置
```
📁 humanoidverse/envs/env_utils/terrain.py
├── curriculum_terrain()                    # 生成课程地形
├── curiculum()                            # 简化版课程地形
└── randomized_terrain()                   # 随机地形(非课程)
```

### 6.2 核心实现
```python
def curriculum_terrain(self):
    """
    生成课程式地形：难度从易到难渐进
    """
    # 按比例分配不同类型的地形
    proportions = np.array(self.cfg.terrain_proportions) / np.sum(self.cfg.terrain_proportions)
    
    # 为每种地形类型分配列范围
    sub_terrain_dict = {}
    for terrain_type in self.cfg.terrain_types:
        # 计算该地形类型的列范围
        start_col, end_col = calculate_col_range(terrain_type, proportions)
        sub_terrain_dict[terrain_type] = (start_col, end_col)
    
    # 生成课程地形
    for terrain_type, (start_col, end_col) in sub_terrain_dict.items():
        for j in range(start_col, end_col):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows  # 难度从0到1渐进
                terrain = self.make_terrain(terrain_type, difficulty)
                self.add_terrain_to_map(terrain, i, j)

def curiculum(self):
    """
    简化版课程地形：行表示难度，列表示类型
    """
    for j in range(self.cfg.num_cols):
        for i in range(self.cfg.num_rows):
            difficulty = i / self.cfg.num_rows      # 行：难度渐进
            choice = j / self.cfg.num_cols + 0.001  # 列：类型变化
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
```

### 6.3 配置参数
```yaml
# humanoidverse/config/terrain/terrain_locomotion.yaml
terrain:
  curriculum: False  # 是否启用课程地形
  num_rows: 10       # 地形行数(难度级别)
  num_cols: 20       # 地形列数(地形类型)
  terrain_types: ["flat", "rough", "stairs", "slopes"]
  terrain_proportions: [0.2, 0.3, 0.3, 0.2]
```

## 🎭 7. 动作采样机制 (Motion Sampling) - **非课程式**

**重要说明**: 经过仔细分析代码，Motion Sampling实际上**不是课程式采样**，而是基于权重的随机采样机制。

### 7.1 实际实现机制
```
📁 humanoidverse/utils/motion_lib/motion_lib_base.py
├── setup_constants()                       # 初始化采样概率
├── load_motions()                          # 基于权重的随机采样
└── _sampling_prob                          # 动作采样权重(目前为均匀分布)
```

### 7.2 真实的采样逻辑
```python
def setup_constants(self):
    """
    初始化动作采样相关的常量
    """
    # 终止历史记录(预留用于未来的课程学习)
    self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
    self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
    self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)

    # 采样概率 - 目前为均匀分布
    self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions

def load_motions(self, random_sample=True, start_idx=0, ...):
    """
    加载动作的核心逻辑
    """
    if random_sample:
        # 基于权重的多项式采样 - 但权重目前是均匀的
        sample_idxes = torch.multinomial(
            self._sampling_prob,
            num_samples=num_motion_to_load,
            replacement=True
        ).to(self._device)
    else:
        # 顺序采样(评估模式)
        sample_idxes = torch.remainder(
            torch.arange(num_motion_to_load) + start_idx,
            self._num_unique_motions
        ).to(self._device)
```

### 7.3 当前状态分析
```python
# 当前的采样权重是均匀分布
self._sampling_prob = torch.ones(num_motions) / num_motions

# 这意味着每个动作被选中的概率相等
# 并没有基于复杂度、成功率或其他指标的权重调整
```

### 7.4 预留的课程学习基础设施
虽然当前不是课程式采样，但代码中预留了相关基础设施：

```python
# 这些变量目前未被使用，但为未来的课程学习做了准备
self._termination_history    # 记录每个动作的终止历史
self._success_rate          # 记录每个动作的成功率
self._sampling_history      # 记录每个动作的采样历史
```

### 7.5 实际的"多样性"机制
```python
def _update_tasks_callback(self):
    """
    定期重新采样动作，增加训练多样性
    这不是课程式的，而是为了避免过拟合特定动作
    """
    if self.config.resample_motion_when_training:
        if self.common_step_counter % self.resample_time_interval == 0:
            self.resample_motion()  # 重新随机采样所有动作
```

### 7.6 总结
- **当前实现**: 均匀随机采样，不是课程式
- **设计意图**: 增加训练多样性，避免过拟合
- **未来潜力**: 代码结构支持添加基于性能的权重调整
- **可能扩展**: 可以基于`_success_rate`或`_termination_history`实现真正的课程式采样

## 📊 8. 课程学习调度机制

### 8.1 更新触发时机
```python
# 在环境重置时更新所有课程
def _reset_tasks_callback(self, env_ids):
    self._episodic_domain_randomization(env_ids)
    
    # 更新各种课程学习
    if self.use_reward_penalty_curriculum:
        self._update_reward_penalty_curriculum()
    
    if self.use_reward_limits_curriculum:
        self._update_reward_limits_curriculum()
    
    if self.add_noise_currculum:
        self._update_obs_noise_curriculum()
    
    # Motion Tracking特有
    if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
        self._update_terminate_when_motion_far_curriculum()
```

### 8.2 课程学习指标
所有课程学习都基于 **平均episode长度** 作为性能指标：
- **episode长度短** → 性能差 → 降低难度
- **episode长度长** → 性能好 → 提高难度

### 8.3 课程学习的通用模式
```python
# 通用课程更新模式
if average_episode_length < level_down_threshold:
    curriculum_value *= (1 ± curriculum_degree)  # 降低难度
elif average_episode_length > level_up_threshold:
    curriculum_value *= (1 ± curriculum_degree)  # 提高难度

curriculum_value = np.clip(curriculum_value, min_limit, max_limit)
```

## 🎯 9. 课程学习的应用效果

### 9.1 训练稳定性
- **渐进式难度**：避免训练初期过难导致的学习失败
- **自适应调整**：根据性能动态调整难度
- **平滑过渡**：避免难度突变导致的性能波动

### 9.2 Sim2Real效果
- **噪声课程**：逐步增加观测噪声，提高鲁棒性
- **限制课程**：动态调整安全限制，平衡性能和安全
- **地形课程**：从简单到复杂地形，提高泛化能力

这个详细的课程式采样分析展示了HumanoidVerse框架中丰富的课程学习机制，这些机制共同作用来提高训练效率和最终性能。
