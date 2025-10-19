# 自适应采样 (Adaptive Sampling) 详细指南

## 概述

自适应采样是BeyondMimic运动跟踪框架中的核心组件，它通过动态调整运动数据的采样策略，优先训练困难的时间片段，从而显著提高训练效率和策略鲁棒性。

## 核心思想

传统的运动跟踪训练通常采用均匀采样策略，即随机选择运动序列中的任意时间点开始训练。然而，这种方法存在以下问题：

1. **训练效率低**: 简单的时间片段被重复训练，浪费计算资源
2. **策略鲁棒性差**: 困难的时间片段训练不足，导致策略在复杂动作上表现不佳
3. **收敛速度慢**: 没有针对性的训练策略，收敛速度较慢

自适应采样通过以下机制解决这些问题：

- **失败跟踪**: 记录每个时间片段的失败历史
- **概率调整**: 根据失败率动态调整采样概率
- **平滑过渡**: 使用核函数确保采样分布的平滑性
- **探索保证**: 保持一定比例的均匀采样

## 代码实现

### 1. 配置参数

```python
@configclass
class MotionCommandCfg(CommandTermCfg):
    """运动命令配置类"""
    
    # 自适应采样参数
    adaptive_kernel_size: int = 1        # 自适应采样核函数大小
    adaptive_lambda: float = 0.8         # 自适应采样衰减因子
    adaptive_uniform_ratio: float = 0.1  # 均匀采样比例 (10%)
    adaptive_alpha: float = 0.001        # 失败计数更新率
```

### 2. 初始化阶段

```python
def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
    """初始化运动命令生成器"""
    super().__init__(cfg, env)
    
    # 加载运动数据
    motion_body_indexes = torch.arange(len(self.cfg.body_names), dtype=torch.long, device=self.device)
    self.motion = MotionLoader(self.cfg.motion_file, motion_body_indexes, device=self.device)
    
    # 计算时间分箱数量
    # 每个时间箱对应一个控制步 (0.02秒)
    self.bin_count = int(
        self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))
    ) + 1
    
    # 初始化失败计数
    self.bin_failed_count = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
    )  # 历史失败计数
    self._current_bin_failed = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
    )  # 当前episode失败计数
    
    # 初始化核函数
    self.kernel = torch.tensor(
        [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], 
        device=self.device
    )
    self.kernel = self.kernel / self.kernel.sum()  # 归一化核函数
```

### 3. 时间分箱计算

```python
# 环境配置参数
decimation = 4          # 控制频率降采样倍数
sim_dt = 0.005          # 仿真时间步长 (秒)
motion_fps = 50         # 运动数据帧率
motion_total_frames = 1943  # 运动数据总帧数

# 计算控制频率
control_freq = 1 / (decimation * sim_dt)  # = 50 Hz

# 计算时间分箱数量
bin_count = int(motion_total_frames // (1 / control_freq)) + 1  # = 97150

# 每个时间箱的详细信息
bin_duration = 1 / control_freq  # = 0.02 秒
bin_frames = bin_duration * motion_fps  # = 1.0 帧
```

### 4. 核心自适应采样算法

```python
def _adaptive_sampling(self, env_ids: Sequence[int]):
    """
    自适应采样方法
    
    根据历史失败情况动态调整运动采样策略，优先采样失败率高的时间片段
    
    Args:
        env_ids: 需要重新采样的环境ID列表
    """
    # 步骤1: 检测失败并更新当前失败计数
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
        # 计算失败时对应的时间箱索引
        current_bin_index = torch.clamp(
            (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 
            0, self.bin_count - 1
        )
        fail_bins = current_bin_index[env_ids][episode_failed]
        # 更新当前失败计数
        self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)
    
    # 步骤2: 计算基础采样概率
    # 失败次数 + 均匀分布比例
    sampling_probabilities = (
        self.bin_failed_count + 
        self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    
    # 步骤3: 应用核函数平滑
    # 使用1D卷积进行概率平滑
    sampling_probabilities = torch.nn.functional.pad(
        sampling_probabilities.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),  # 非因果核
        mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
        sampling_probabilities, 
        self.kernel.view(1, 1, -1)
    ).view(-1)
    
    # 步骤4: 归一化概率分布
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
    
    # 步骤5: 多项式采样
    sampled_bins = torch.multinomial(
        sampling_probabilities, 
        len(env_ids), 
        replacement=True
    )
    
    # 步骤6: 转换为具体时间步
    # 添加随机偏移避免固定采样点
    self.time_steps[env_ids] = (
        (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
        / self.bin_count
        * (self.motion.time_step_total - 1)
    ).long()
    
    # 步骤7: 计算采样指标
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count)
    pmax, imax = sampling_probabilities.max(dim=0)
    
    # 记录采样指标
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
```

### 5. 失败计数更新机制

```python
def _update_command(self):
    """每个时间步更新命令"""
    self.time_steps += 1
    
    # 检查哪些环境需要重新采样
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    self._resample_command(env_ids)
    
    # 更新失败计数 (指数移动平均)
    self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed + 
        (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
    )
    self._current_bin_failed.zero_()  # 清零当前计数
```

### 6. 重新采样触发

```python
def _resample_command(self, env_ids: Sequence[int]):
    """重新采样运动命令"""
    if len(env_ids) == 0:
        return
    
    # 执行自适应采样
    self._adaptive_sampling(env_ids)
    
    # 重新设置运动命令
    # 包括位置、姿态随机化等
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()
    
    # 应用随机化
    range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
    
    root_pos[env_ids] += rand_samples[:, 0:3]
    root_ori[env_ids] = quat_mul(root_ori[env_ids], euler_xyz_to_quat(rand_samples[env_ids, 3:6]))
    # ... 更多随机化逻辑
```

## 关键参数详解

### 1. 时间分箱参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `bin_count` | 97,150 | 总时间箱数量 |
| `bin_duration` | 0.02秒 | 每个时间箱的时长 |
| `bin_frames` | 1.0帧 | 每个时间箱对应的运动帧数 |

### 2. 自适应采样参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `adaptive_kernel_size` | 1 | 核函数大小（当前为1，不进行平滑） |
| `adaptive_lambda` | 0.8 | 核函数衰减因子 |
| `adaptive_uniform_ratio` | 0.1 | 均匀采样比例（10%） |
| `adaptive_alpha` | 0.001 | 失败计数更新率 |

### 3. 采样指标

| 指标 | 说明 |
|------|------|
| `sampling_entropy` | 采样分布的熵，衡量采样多样性 |
| `sampling_top1_prob` | 最高概率时间片的概率值 |
| `sampling_top1_bin` | 最高概率对应的时间箱位置 |

## 算法流程

```
1. 初始化
   ├── 计算时间分箱数量
   ├── 初始化失败计数
   └── 初始化核函数

2. 每个时间步
   ├── 时间步递增
   ├── 检查是否需要重新采样
   └── 更新失败计数

3. 重新采样触发
   ├── 检测失败环境
   ├── 更新当前失败计数
   └── 执行自适应采样

4. 自适应采样
   ├── 计算基础采样概率
   ├── 应用核函数平滑
   ├── 归一化概率分布
   ├── 多项式采样
   ├── 转换为时间步
   └── 计算采样指标
```

## 优势分析

### 1. 训练效率提升

- **针对性训练**: 重点训练困难时间片段
- **资源优化**: 避免重复训练简单片段
- **收敛加速**: 更快达到理想性能

### 2. 策略鲁棒性增强

- **全面覆盖**: 确保所有时间片段都得到充分训练
- **困难优先**: 优先解决复杂动作问题
- **平滑过渡**: 核函数确保相邻时间片的平滑性

### 3. 自适应特性

- **动态调整**: 根据训练进度自动调整采样策略
- **探索保证**: 10%均匀采样确保不会忽略某些区域
- **历史记忆**: 通过指数移动平均保持历史信息

## 实际效果

从训练日志可以看到自适应采样的效果：

```
Metrics/motion/sampling_entropy: 0.2979    # 采样熵，表示分布多样性
Metrics/motion/sampling_top1_prob: 0.0561  # 最高概率时间片的概率
Metrics/motion/sampling_top1_bin: 0.5641   # 最高概率对应的时间箱位置
```

这些指标表明：

1. **采样多样性适中**: 熵值0.2979表示采样分布既不过于集中也不过于分散
2. **概率分布合理**: 最高概率0.0561表明没有过度集中在某个时间片
3. **时间分布均匀**: 最高概率时间箱在0.5641位置，分布相对均匀

## 总结

自适应采样是BeyondMimic框架的核心创新之一，它通过智能的采样策略显著提升了运动跟踪的训练效率和策略质量。通过动态调整采样概率，系统能够自动识别和重点训练困难的时间片段，从而在保持训练稳定性的同时大幅提升性能。

这种设计使得机器人能够更好地学习复杂的舞蹈动作，实现高质量的全身运动跟踪，为实际部署奠定了坚实的基础。
