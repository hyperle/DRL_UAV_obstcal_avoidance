# 双目深度四旋翼避障：训练与部署骨架

当前仓库提供第一阶段研究 MVP：Gymnasium 深度避障环境、SB3 PPO 训练/评估、策略 ONNX 导出及一致性校验、ROS 2 C++ 推理节点。ROS 2 端采用抽象 `FlightControllerAdapter`，不绑定 PX4、ArduPilot 或具体飞控，也不输出电机 PWM。

## 当前策略接口

| 张量 | 形状 | 含义 |
|---|---:|---|
| `depth` | `[1, 1, 32, 48]` float32 | 深度米数裁剪至 `[0, 8]` 后除以 8；无效/无返回值映射为 1 |
| `state` | `[1, 11]` float32 | 目标机体系位置 3、机体系速度 3、偏航角速度 1、上一步归一化动作 4 |
| `action` | `[1, 4]` float32 | 归一化机体系前向/左向/上向速度和偏航角速度，范围 `[-1, 1]` |

ROS 2 状态向量缩放和动作缩放必须与 `uav_rl/envs/depth_avoidance.py` 中的常量一致。部署输入是已生成的单通道深度图，不是左右目原始图像；双目匹配/深度生成由相机驱动或独立感知节点负责。

## Python 训练

建议在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[export,test]'
```

也可以使用仓库启动器。首次运行会自动创建 `.venv` 并安装依赖；不带子命令时默认开始 1,000,000 步 PPO 训练：

```bash
./run-drl
./run-drl test
./run-drl train --timesteps 200000 --n-envs 8
./run-drl evaluate artifacts/ppo_depth_avoidance.zip --episodes 100
./run-drl export artifacts/ppo_depth_avoidance.zip
./run-drl all --timesteps 200000
```

脚本实际位于 `.script/run-drl`，根目录的 `run-drl` 是便捷入口。若希望在任意目录直接输入裸命令 `run-drl`，将仓库根目录加入 shell 的 `PATH`，例如：

```bash
export PATH="$PWD:$PATH"
```

运行检查和短训练：

```bash
pytest
uav-train --timesteps 200000 --n-envs 8 --output artifacts/ppo_depth_avoidance
uav-evaluate artifacts/ppo_depth_avoidance.zip --episodes 100
uav-export-onnx artifacts/ppo_depth_avoidance.zip --output artifacts/depth_policy.onnx
```

`DepthAvoidanceEnv` 是快速的理想化几何投影环境：球形障碍物投影到低分辨率深度图，动力学采用一阶速度响应。它适合先验证 MDP、奖励、动作接口和训练代码，不代表真实双目成像，也尚未校准到 240g 机体。进入实机前，应接入含相机基线、内外参、飞控响应、时延和图像噪声的仿真/SITL 环境并重新验证策略。

## ROS 2 C++ 部署

先安装 ROS 2、`cv_bridge`、OpenCV 以及与目标机架构匹配的 ONNX Runtime C++ 包。示例接受 `32FC1` 米或 `16UC1` 毫米深度图，并订阅 `nav_msgs/Odometry` 与同一世界坐标系的 `PointStamped` 目标。

```bash
cd ros2_ws
colcon build --packages-select stereo_avoidance \
  --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

将训练导出的模型复制到部署目录并配置 `model_path`、topic 和 frame。默认 ROS topic adapter 发布 `TwistStamped`，命令按 `base_link` 的 FLU 约定表示。硬件适配器必须将 FLU/世界坐标约定转换为目标飞控要求，并实现其 offboard、使能、看门狗和安全停止语义。示例中的零速度 `hold()` 只是接口占位，不能视为任何飞控上的通用安全动作。

节点会在输入未齐、输入过期、坐标系不匹配或推理异常时调用 `hold()`。上线前还必须依据实际相机频率和控制链路调整超时/推理频率，并加入限速、地理围栏、人工急停及飞控自身 failsafe。

## 后续研究里程碑

1. 用实机标定的相机、速度/姿态数据和飞控响应替换理想化环境参数。
2. 接入 Gazebo/SITL，逐步随机化障碍物、光照、深度噪声、时延、风扰、质量/惯量和推力响应。
3. 报告成功率、碰撞率、最小间距、路径代价、控制平滑性、人工接管率和目标板端推理 P95 时延；开展奖励与输入表征消融。
4. 在无桨、仿真、保护网/限速等分级条件下验证；未完成适配器和安全审查前不要接真实电机控制。

## 1. 环境交互

  实现位置：

  uav_rl/envs/depth_avoidance.py:32

  环境继承 Gymnasium：

  class DepthAvoidanceEnv(gym.Env):

  ### 观测空间

  在 observation_space 中定义：

  {
      "depth":  [1, 32, 48],
      "state":  [11]
  }

  对应：

  - depth：模拟双目相机生成的深度图；
  - state：目标相对位置、无人机速度、偏航角速度、上一时刻动作。

  代码位置：

  uav_rl/envs/depth_avoidance.py:69

  ### 动作空间

  动作是 4 维连续值：

  [forward_velocity,
   lateral_velocity,
   vertical_velocity,
   yaw_rate]

  范围为 [-1, 1]，再映射为实际速度和角速度。

  代码位置：

  uav_rl/envs/depth_avoidance.py:85

  ### 环境初始化

  reset() 会完成：

  - 随机生成目标点；
  - 随机生成障碍物；
  - 初始化无人机位置、速度和姿态；
  - 设置动力学随机化参数；
  - 返回初始深度图和状态。

  代码位置：

  uav_rl/envs/depth_avoidance.py:103

  ### 单步交互

  step(action) 完成一次完整的环境交互：

  1. 接收策略输出动作；
  2. 将归一化动作映射为速度指令；
  3. 更新无人机速度、位置和偏航角；
  4. 计算目标距离；
  5. 判断碰撞、越界和成功；
  6. 生成深度图；
  7. 计算奖励；
  8. 返回下一时刻观测。

  代码位置：

  uav_rl/envs/depth_avoidance.py:138

  典型交互过程相当于：

  observation, info = env.reset()

  while True:
      action = policy(observation)
      next_observation, reward, terminated, truncated, info = \
          env.step(action)

      if terminated or truncated:
          break

      observation = next_observation

  深度图由 _get_depth() 根据球形障碍物的几何位置投影生成：

  uav_rl/envs/depth_avoidance.py:244

  最终观测由 _observation() 组合：

  uav_rl/envs/depth_avoidance.py:303

  ## 2. 网络训练

  训练入口是：

  uav_rl/training.py

  ### 并行创建环境

  vector_env = make_vec_env(
      DepthAvoidanceEnv,
      n_envs=n_envs,
      seed=seed,
      env_kwargs={"domain_randomization": True},
  )

  代码位置：

  uav_rl/training.py:66

  例如默认 n_envs=8 时，会同时运行 8 个独立环境，加快经验采样。

  ### 网络结构

  视觉输入和状态输入使用双分支网络：

  深度图
    -> Conv2d
    -> Conv2d
    -> Conv2d
    -> Flatten
    -> 视觉特征

  状态向量
    -> Linear
    -> 状态特征

  视觉特征 + 状态特征
    -> Fusion Layer
    -> Actor/Critic

  自定义特征提取器是：

  class DepthStateExtractor(BaseFeaturesExtractor):

  代码位置：

  uav_rl/training.py:17

  前向计算在：

  uav_rl/training.py:42

  ### PPO 算法

  训练采用 Stable-Baselines3 的 PPO：

  model = PPO(
      "MultiInputPolicy",
      vector_env,
      policy_kwargs=build_policy_kwargs(),
      ...
  )

  代码位置：

  uav_rl/training.py:72

  PPO 会自动完成：

  1. 用当前策略向环境采样；
  2. 获取 (observation, action, reward, done)；
  3. 保存 rollout；
  4. 计算回报和优势函数；
  5. 更新 actor；
  6. 更新 critic；
  7. 重复上述过程。

  真正开始训练的位置是：

  model.learn(
      total_timesteps=timesteps,
      progress_bar=False,
  )

  代码位置：

  uav_rl/training.py:87

  因此当前不是手写 PPO 更新，而是：

  自定义 Gymnasium 环境
          ↓
  SB3 自动调用 reset()/step()
          ↓
  PPO 采样轨迹
          ↓
  CNN + MLP 策略网络
          ↓
  梯度更新
          ↓
  保存 actor/critic 模型

  训练完成后保存：

  model.save(str(output))

  ## 3. 当前实现的边界

  目前的“环境交互”是 Python 内部的快速仿真，包含：

  - 简化无人机运动学；
  - 障碍物几何模型；
  - 模拟深度图；
  - 奖励和终止逻辑。

  但它还不是：

  - Gazebo/SITL 真实飞行动力学；
  - ROS 2 传感器闭环训练；
  - 真实双目视觉匹配；
  - 真实飞控硬件交互。

  当前完整流程是：

  DepthAvoidanceEnv
      ↓
  PPO 采样和训练
      ↓
  导出 actor 为 ONNX
      ↓
  ROS 2 C++ 节点加载 ONNX
      ↓
  接收真实深度图/里程计
      ↓
  发布速度控制指令

  所以，当前代码已经实现了研究原型所需的“环境—智能体—奖励—训练”闭环；下一阶段需要把这个理想化环境替换或并行接入 Gazebo/SITL，才能实现更接近真实无人机的环境交互。