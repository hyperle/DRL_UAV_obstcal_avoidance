# 深度避障工程性能分析

本文分析当前仓库中 Python 训练环境、PPO 策略、ONNX/C++ 推理节点和 ROS 2 数据链路的性能。结论分为两类：带有“实测”的内容来自当前开发机；其余是根据代码结构得到的静态分析，不能替代目标机（尤其是 K230 以上实际平台）的端到端测量。

## 1. 测试条件与基准结果

本次基准使用当前 `.venv` 中的依赖版本：

| 项目 | 值 |
|---|---|
| CPU | 13th Gen Intel Core i5-13500H，16 logical CPUs |
| Python | 3.12 |
| NumPy | 2.5.3 |
| Gymnasium | 1.3.0 |
| PyTorch | 2.14.1+cu130 |
| Stable-Baselines3 | 2.9.0 |
| 环境设置 | `DepthAvoidanceEnv()`，12 个球形障碍物，默认深度噪声和域随机化 |
| 测试方式 | 单环境连续执行 10,000 次 `step()`，遇到 episode 结束立即 reset |

实测结果：

```text
elapsed = 1.6107 s
throughput = 6,208.5 environment steps/s
latency = 0.161 ms/step
```

这只是环境模拟速度，不是 PPO 训练速度，也不是部署端推理时延。此前 2 个并行环境、2,048 步的 PPO 冒烟训练成功，日志显示约 2,344 rollout steps/s；该值包含策略前向、采样、优势计算和更新，不能直接外推到长训练或其他硬件。

## 2. Python 环境开销

### 2.1 每步主要计算

`DepthAvoidanceEnv.step()` 的计算量与障碍物数量 `N` 近似线性相关：

1. 速度响应、坐标变换、目标距离和边界判断：常数级。
2. `_min_clearance()`：对全部障碍物计算一次三维距离，`O(N)`。
3. `_get_depth()`：再次遍历全部障碍物，并对每个障碍物做投影、切片和 `np.minimum`，平均为 `O(N + P)`；`P` 是投影覆盖的像素数量。
4. 深度噪声：对固定的 `32×48=1536` 个像素生成随机数，成本与障碍物数量无关。
5. 观测拼接和字典创建：固定的小额分配开销。

当前默认 `N=12`、分辨率 `32×48`，所以环境非常轻量；6,208 step/s 足以支撑初期 PPO 采样。若提高障碍物数量、深度分辨率或改为真实双目渲染，瓶颈会从 NumPy 几何投影转移到渲染、立体匹配和图像搬运。

### 2.2 当前实现中的分配与优化空间

- 每一步都会创建动作裁剪结果、世界速度数组、深度图数组、噪声数组、状态拼接数组和 observation 字典。
- `_get_depth()` 每次从全量 `MAX_DEPTH_M` 图开始，并对障碍物执行 Python 层循环；这便于研究但不适合极大规模并行环境。
- `_sample_obstacles()` 在 reset 时最多尝试每个障碍物 100 次。障碍物密度很高时，reset 时间可能显著增加。
- 训练中的 `info` 包含多项 Python 标量和布尔值，便于日志记录，但高并发采样时会增加对象创建开销。

优化优先级：

1. 先测量真实训练瓶颈；不要在当前 0.161 ms/step 已足够时过早重写环境。
2. 需要更高采样率时，预分配深度缓冲区和状态缓冲区，减少每步 NumPy 对象分配。
3. 将障碍物投影改成批量 NumPy 运算，或使用 Numba/JAX/C++ 扩展；保持观测结果和随机种子可复现。
4. 对真实视觉仿真使用 GPU/并行渲染，而不是继续优化这个理想化投影器。

## 3. PPO 策略与训练性能

当前视觉特征提取器为：

```text
depth [1, 32, 48]
  Conv2d(1,16,k=5,s=2) -> ReLU
  Conv2d(16,32,k=3,s=2) -> ReLU
  Conv2d(32,64,k=3,s=2) -> ReLU -> Flatten
state [11] -> Linear(11,32) -> ReLU
拼接 -> Linear(...,128) -> ReLU
actor: 128 -> 128 -> 64 -> 4
critic: 128 -> 128 -> 64 -> 1
```

该网络输入只有 1,547 个观测元素（1,536 个深度值加 11 个状态值），分辨率和通道数都很小，部署推理计算量预计远小于深度立体匹配和相机预处理。实际参数量应在导出后用 Netron、ONNX Runtime profiling 或脚本统计确认；参数量不是端到端时延的唯一决定因素。

当前训练配置：

- PPO `n_steps=1024`、`batch_size=256`、`gamma=0.99`、`gae_lambda=0.95`。
- 默认 `n_envs=8`，每个 rollout 收集约 8,192 个环境步。
- `torch.set_num_threads(1)`，避免每个 Python worker/环境内部的 CPU 线程过度竞争。
- `make_vec_env` 使用同步向量环境；环境本身是 Python 代码，因此 `n_envs` 增加后不会线性扩展。

训练速度应报告为 `environment steps/s` 和总 wall-clock 时间，而不是只报告迭代次数。建议分别测试 `n_envs ∈ {1, 2, 4, 8, 16}`，寻找采样和策略更新的拐点；并固定随机种子、timesteps、设备和线程数。

## 4. C++/ONNX Runtime 部署链路

当前每个策略周期的链路是：

```text
深度 Image 回调
  -> cv_bridge 视图/16UC1 转 32FC1
  -> OpenCV resize 到 32×48
  -> 无效值处理、裁剪、归一化、vector 分配
  -> 加锁保存最新 depth

定时器回调
  -> 加锁并复制 depth/odometry/goal
  -> 时间戳和 frame 检查
  -> 构造 11 维 state
  -> Ort::Value 创建两个 CPU tensor
  -> session->Run()
  -> 动作限幅、速度尺度映射
  -> 发布 TwistStamped
```

### 4.1 主要瓶颈

1. **深度预处理分配**：每张图都会创建 `cv::Mat`/resize 结果和 1,536 元素的 `std::vector<float>`。在 15 Hz 时通常可接受，但高帧率下会增加 allocator 和缓存压力。
2. **定时器回调复制**：策略周期会复制深度 vector、完整 `Odometry` 和 `PointStamped`。深度复制很小（约 6 KiB），但仍可通过预分配/双缓冲减少抖动。
3. **ONNX tensor 与输出分配**：`DepthPolicy::infer()` 每次创建 `Ort::Value` 和输出 vector。应使用 ONNX Runtime I/O binding、复用输入缓冲区，或至少复用固定 shape 的工作区。
4. **回调串行化**：默认 ROS 2 executor 中，图像预处理、策略推理和发布可能在同一执行线程上运行；如果推理偶发超时，图像回调会排队或丢弃，控制输出仍需依靠 stale timeout 和飞控 watchdog。
5. **模型/预处理一致性**：部署端将无效深度映射为最大距离、裁剪到 8 m、缩放到 `[0,1]`，必须与训练定义完全一致，否则“性能好”也可能只是输入分布错位。

### 4.2 实时性目标

当前节点默认策略频率为 15 Hz，周期约 66.7 ms；输入过期阈值为 250 ms。建议将目标板验收指标定义为：

| 指标 | 初始目标 |
|---|---:|
| 策略端到端 P50 | < 15 ms |
| 策略端到端 P95 | < 30 ms |
| 策略端到端 P99 | < 50 ms |
| 连续周期超时 | 0 |
| 深度/里程计 stale 触发 | 仅在输入确实过期时触发 |

这些是工程起始目标，不是对未知目标板的保证。必须在实际机载计算平台上测量从 ROS 消息接收至 `TwistStamped` 发布的完整路径，并包括相机驱动、深度生成、进程调度和日志关闭后的情况。

## 5. 发现的设计风险

- **当前环境不是飞行动力学仿真**：一阶速度响应没有表示姿态环、推力饱和、惯量、气动、风扰、执行器延迟或电池电压变化。环境速度很快不能说明策略可直接上机。
- **奖励使用仿真真实障碍几何**：策略只看到深度图，但奖励中的最近间隙来自真实障碍物状态；这是训练时常见的 privileged reward，部署时不可直接获得，必须确认没有把真实状态泄漏进观测或控制。
- **深度图是单通道理想输入**：真实双目匹配会有遮挡、纹理缺失、远距离噪声、边缘孔洞和时间同步误差。应在仿真中随机化这些因素。
- **状态坐标和控制坐标必须固定**：当前动作约定为机体系 FLU 速度和偏航角速度；硬件适配器必须明确 ENU/NED、机体系轴向、单位、offboard watchdog 和 failsafe。
- **stale timeout 不是完整安全机制**：零速度 `hold()` 只是通用占位。真实飞控必须实现经验证的悬停、制动、降落或交还控制权流程。
- **Torch/Python 版本会影响可复现性**：安装器使用开放版本范围，未来依赖升级可能改变性能或 ONNX 导出行为。研究实验应记录完整 `pip freeze`、commit、随机种子和设备信息。

## 6. 推荐的性能测试脚本与记录格式

每个实验至少记录：

```text
git revision
Python / NumPy / PyTorch / SB3 / ONNX Runtime versions
CPU/GPU/NPU model and power mode
environment: obstacle_count, depth size, noise, domain randomization
training: n_envs, n_steps, batch_size, total_timesteps, device, torch threads
deployment: image rate, policy rate, queue depth, executor type
P50/P95/P99 latency, steps/s, dropped frames, stale events, CPU/GPU utilization
```

建议将性能拆成四个独立基准：

1. 仅 `env.reset/step`。
2. 仅 PyTorch actor 前向。
3. ONNX Runtime `session->Run()`，固定输入。
4. ROS 2 真实消息链路端到端。

这样才能判断瓶颈究竟来自环境、策略、运行时、深度生成还是 ROS 2 调度。

## 7. 结论

当前 MVP 的 Python 几何环境性能良好，单环境约 6,208 step/s；网络输入小、结构轻，适合在更高算力平台上继续研究。现阶段真正的性能不确定性在部署侧：双目深度生成、OpenCV 预处理、ONNX Runtime 首次/稳态时延、ROS 2 回调调度以及飞控链路。进入实机前，应优先完成目标板端到端 profiling，再决定是否需要量化、I/O binding、预分配、独立 callback group 或降低输入分辨率/策略频率。
