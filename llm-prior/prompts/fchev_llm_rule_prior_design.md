你是一名熟悉强化学习、Soft Actor-Critic（SAC）、车辆纵向控制和燃料电池混合动力汽车（FCHEV）能量管理的工程师。请根据下述项目接口，为 FCHEV 环境设计一个可执行的 Python rule-prior 程序。该程序为两个独立 SAC agent 提供高斯动作分布先验。它不是独立控制器，也不能绕过环境物理约束。

### 1. 项目背景与优化目标

本项目研究燃料电池混合动力汽车的驾驶与能量管理策略。SUMO 仿真道路、引导车辆、交通信号灯和车辆运动；Python 环境将车辆纵向动力学与燃料电池系统（FCS）、电池和电驱单元（EDU）模型耦合起来。

控制器由两个相互独立的随机 SAC agent 组成，各自使用独立的 observation 和 reward：

- **Driving agent：** 一级且占主导地位的任务是跟随引导车辆。它使用当前 ego 和 lead 的速度、加速度、车距，以及下文列出的信号灯和限速信息，决定 ego 加速度。只要可行，就应保持不小于 `D_safe` 的安全车距，并尽量避免车距超过规定的最大距离 `D_max`；车距过大时应采取行动缩短距离。车距小于 `D_safe` 属于过近，存在碰撞或车辆重叠风险。跟车始终是驾驶的主要目标，同时不得违反当前道路区域的限速。ego 只能在绿灯时通过信号灯路口，必须在绿灯时越过停止线；红灯和黄灯都禁止越线。
- **Energy agent：** 选择 FCS 请求输出功率。其目标是降低氢耗、FCS/电池/EDU 退化，并使电池 SOC 接近目标值 0.5。

两个 agent 分别优化各自对应的 reward。Driving agent 的主要目标是在安全车距和规定的最大车距范围内跟车，同时遵守限速和交通信号；energy agent 负责功率管理、部件健康和 SOC 调节。两者的 reward 不会合并为共享 team reward。

控制周期为每步 1 s。每一步中，两个 agent 都根据该时刻的当前状态分别决策。Energy agent 的 observation 包含当前测得的 ego 加速度和 ego 速度，不包含 driving agent 在本步新选择的加速度。Rule program 每步基于同一份完整原始 observation 调用一次，同时生成两个 prior。两项 action 在同一个环境 step 中执行；下一步根据更新后的状态重新决策。

SOC 偏差属于能量管理目标，不是电池退化指标。当前 observation 不直接包含电机扭矩、瞬时 EDU 损耗、请求 FCS 功率、电池功率或氢耗。不要假设 rule 能直接获得这些量。如果需要估算，必须将其明确标注为近似并说明局限。

### 2. 当前仿真环境与精确数值

**道路与任务：**

- 路线包含 25 个信号灯路口，路口间距 1000 m，相邻路口之间有 24 段道路。每个 episode 最长 6000 s，每个控制 step 为 1 s。
- Ego 和 lead 均以 0 m/s 起步，初始车距为 10 m。Lead 的速度和加速度随场景变化，并会在路线终点停车。
- **主要驾驶任务——跟车：** Driving agent 的主要任务是跟随 lead。使用当前 ego、lead 的速度和加速度，以及观测到的车距，在可行时将 ego 维持在安全距离 `D_safe` 和规定最大距离 `D_max` 之间。若车距小于 `D_safe`，应降低接近速度并保护最小安全距离。若红绿灯、限速或其他限制导致车距超过 `D_max`，应将其视为追赶情形：在继续遵守限速和绿灯通行要求的同时，采取可行行动主动缩小车距，使其回到 `D_safe`–`D_max` 范围。受限时仍应向 lead 推进；不要因为车距已经过大就保持停车。
- **决策频率和预测窗口：** 每个 1 s step 都使用当步 observation 调用一次 rule。每次决策使用有限预测窗口 `H=5–20` steps（5–20 s），预测 ego、lead、信号灯、限速区域和近似功率需求的可能结果。Driving 加速度 prior 和 energy 功率 prior 都要参考这个相同窗口内的预测结果。由于 SUMO 在单步内的运动和积分方式可能与简化预测不同，驾驶预测须加入 1 s 的保守不确定裕量：考虑 ego 实际行驶距离可能比名义预测多一个 step，并在评估跟车间距、越过停止线和进入限速区域时使用这个更早到达的情况。不要预测超过 `H` 的未来；下一 step 根据新 observation 重新计算 prior。
- **Action 范围和物理能力：** Driving agent 请求加速度范围为 `[-1.5,+0.7] m/s²`（最大请求制动减速度 1.5 m/s²，最大正加速度 0.7 m/s²）。Energy agent 请求 FCS 输出功率范围为 `[0,50] kW`。Ego 另有 70 km/h 的硬性最高速度，环境禁止倒车。电池、EDU 和动力系统可行域还可能进一步限制实际加速度或交付功率。
- **限速：** 普通道路限速为 60 km/h（16.6667 m/s）。每个停止线前 100 m 起至路口内部，ego 限速为 30 km/h（8.3333 m/s）；驶离路口限速区后恢复 60 km/h。Observation 提供当前限速、下一个限速以及到下一个限速区域入口的距离。红灯不会把限速 observation 改成 0。
- **交通信号灯：** 每个信号灯周期为 90 s：绿灯 40 s、黄灯 3 s、红灯 47 s。各路口信号相位有不同偏移。Observation 提供下一个信号灯状态、到停止线距离以及当前相位剩余时间。Ego 只能在绿灯时越过停止线；红灯和黄灯都禁止越线。若预测某个候选 action 会在 `H` 内到达停止线，应根据预测车辆运动和越线时的信号相位判断该 action 是否允许。不要在 `H` 之外进行无界的信号预测；下一 step 重新判断。
- **跟车与道路规则的平衡：** 跟车是首要驾驶目标。根据同一个预测窗口，在跟车的基础上评估限速和信号灯要求。预测结果显示存在冲突时，应在尽量维持跟车的同时遵守道路限速和绿灯越线要求。若这些约束使目标车距暂时无法实现，应先遵守道路和信号规则，并在条件允许后恢复跟车。
- Lead 到达路线终点并基本静止时 episode 正常结束；Ego 完成路线也会结束 episode。

**车辆与 action 范围：**

- 车辆质量 7000 kg；长度 8 m；车轮半径 0.4 m；迎风面积 5.8 m²；风阻系数 0.55；滚动阻力系数 0.0085；旋转质量系数 1.05；空气密度 1.225 kg/m³；重力加速度 9.8 m/s²；道路坡度 0°。
- 归一化驾驶 action `u_d ∈ [-1,1]` 映射为请求加速度：若 `u_d < 0`，`a_req = 1.5u_d`（[-1,0] 映射到 [-1.5,0] m/s²）；若 `u_d ≥ 0`，`a_req = 0.7u_d`（[0,1] 映射到 [0,0.7] m/s²）。
- 环境强制速度不小于零：控制步长为 1 s 时，加速度不会低于 `−v/Δt`，所以车辆静止时负加速度请求会被截为 0 m/s²；车辆行驶时可以减速至静止，但不会倒车。不要在 rule 中额外设置低速阈值，把负加速度请求改为零；不倒车限制由环境处理。
- Ego 的 SUMO speed mode 为 0，环境直接设置 ego 速度。因此 SUMO 不会监督 ego 跟车、限速或红灯停车。道路限速由 reward/终止逻辑处理；70 km/h 车辆最高速度则是独立硬约束。
- 归一化能量 action `u_e ∈ [-1,1]` 映射为 FCS 请求功率 `P_req = 25(u_e+1) kW`，即 [0,50] kW。
- FCS 最大输出为 50 kW，DC/DC 效率为 0.95。实际 FCS 输出可能受能量调度、EDU 需求和电池物理限制影响，不保证等于请求值。
- 电池模型为 A123 26650 LiFePO₄ 电芯组成的 106 串 134 并电池包。单体标称电压 3.3 V、容量 2.3 Ah。初始和目标 SOC 均为 0.5；SOC 范围 [0.15,0.9]；最大放电和充电倍率分别为 2C、1C；单体电压范围 2.0–3.6 V；初始 core、surface、coolant 温度均为 25°C。电池使用二阶 RC 电热模型和查表数据。
- EDU 使用 EPA Generic IPM 150 kW EDU 完整 map，物理边界包含电机、逆变器和变速箱，并非单独电机 map。齿轮比 9.5。数据集没有提供额定效率，不要把峰值效率称为额定效率。环境使用完整 EDU 效率/损耗 map 和有效工作区间，不要重复加入逆变器或变速箱损耗。
- FCS 氢耗表来自 Lin and Zheng (2011) 原始 50 kW FCS 多项式，没有缩放到其他额定功率。退化模型考虑启动、低功率、高功率和负载变化退化。EDU 老化参考点为机械功率 75 kW、扭矩 156 Nm、转速 4591 rpm，参考寿命 30000 h。电池 SOH 由当前电热和老化模型更新。
- Ego 没有跟车或信号灯监督器。SUMO 不会自动控制 ego 跟随 lead，也不会在红灯时自动让 ego 停车。电池/FCS/EDU 可行域、不倒车规则和最高速度硬限制仍然有效。物理约束可能使实际执行 action 与请求不同。

### 3. Observation、Agent 输入与量纲

Rule program 接收**完整、原始、未归一化的 15 维 observation**，顺序如下。当没有下一个信号灯时，环境将其状态编码为绿灯（1）、距离 30000 m、相位剩余时间 0 s。30000 m 是“无可见目标/超出有效范围”的哨兵值。信号灯距离 0 m 仍是有效 observation，表示 ego 位于停止线或已到达停止线；不表示没有信号灯。除非接口明确规定，否则不要将 0 或负距离解释为信号灯不存在。下一个信号灯可能在较远处可见；其相位相关性每步根据预计到达时间和当前相位剩余时间评估，与 100 m 限速区域独立。

| 索引 | 名称 | 单位/编码 |
|---:|---|---|
| 0 | 电池 SOC | 比例值，通常 [0.15,0.9] |
| 1 | 电池 SOH | 比例值，初始值 1 |
| 2 | FCS SOH | 比例值，初始值 1 |
| 3 | EDU SOH | 比例值，初始值 1 |
| 4 | 当前 ego 加速度 | m/s² |
| 5 | 当前 ego 速度 | m/s |
| 6 | 当前 lead 加速度 | m/s² |
| 7 | 当前 lead 速度 | m/s |
| 8 | Ego 到 lead 车距 | m，观测最大值 30000 |
| 9 | 下一个信号灯状态 | 红 0、绿 1、黄 2 |
| 10 | 到下一个信号灯停止线的距离 | m，最大 30000；无信号灯时为 30000 |
| 11 | 当前信号相位剩余时间 | s，通常 [0,100]；无信号灯时为 0 |
| 12 | 当前区域限速 | m/s；主路 16.6667，路口限速区 8.3333 |
| 13 | 下一个限速区域的限速 | m/s |
| 14 | 到下一个限速区域入口的距离 | m，最大 30000 |

当前 SAC 实验配置开启 observation normalization。筛选各 agent 输入前，SOH 索引 1–3 变为 `(1−SOH)×1000`；加速度索引 4、6 除以 1.5；速度索引 5、7 除以 20；车距/距离索引 8、10、14 除以 1000；信号灯状态索引 9 除以 2；相位时间索引 11 除以 100；限速索引 12、13 除以 20。Driving agent 接收归一化后的索引 4–14（11 个值）。Energy agent 接收归一化后的索引 [4,5,0,1,2,3]（6 个值）：当前 ego 加速度、ego 速度、SOC、电池 SOH、FCS SOH 和 EDU SOH。在同一个 step 中，两个 agent 的输入都来自 action 执行前的同一个环境 observation；Energy agent 不会收到 Driving agent 本步刚选出的 action。Rule program 始终接收完整的 15 维原始 observation。

### 4. SAC Action 与物理执行

每个独立 SAC actor 都表示一个 latent 空间中的一维高斯策略。Actor 预测 mean 和 log standard deviation；`log_std` 截断到 [-20,2]，`std = exp(log_std)`。随机策略采样为 `z = mean + std × ε`，其中 `ε ~ Normal(0,1)`，再计算 `u = tanh(z)`。确定性评估使用 `u = tanh(mean)`。

- Driving actor：`u_d ∈ [-1,1]`，再由第 2 节的分段映射转换为 [-1.5,0.7] m/s² 的请求加速度。
- Energy actor：`u_e ∈ [-1,1]`，再由 `P_req = 25(u_e+1) kW` 转换为 [0,50] kW 的 FCS 输出功率请求。
- 两个 actor 相互独立，使用不同 observation 和 reward。任何一个 actor 的输入都不含另一个 actor 本步新选出的 action。
- 环境随后检查车辆动力学和动力系统可行性。最高速度、不倒车、电池功率/SOC/电流/电压限制、EDU 扭矩-转速/效率有效域及 FCS 限制都可能截断或改变请求。Rule prior 不能绕过或修改这些约束。

### 5. 当前 Reward 定义

两个独立 SAC agent 使用不同 reward。下表数值对应 `config/experiments/maddpg_individual_reward_g0999_500ep.json` 这一独立 reward 参考配置，不保证其他实验配置使用相同数值。Driving agent 优化 driving reward，energy agent 优化 energy reward；两者不会收到共享 team reward。

| 系数 | 当前值 | 主要用途 |
|---|---:|---|
| β₀ | 1 | 跟车 |
| β₁ | 0.2 | 氢耗 |
| β₂ | 5000 | FCS 退化增量 |
| β₃ | 100000 | 电池 SOH 损失 |
| β₄ | 100000 | EDU 退化增量 |
| β₅ | 2 | 驾驶舒适性 |
| β₆ | 10 | SOC 偏差 |
| β₇ | 1 | 红绿灯预测 reward |
| β₈ | 500 | 进入低限速区域前的速度 shaping |

这些 reward 函数和系数是 RL 的训练信号，用于解释 actor 被鼓励学习的行为。生成的 rule 是参考 prior，不是 reward 函数的代码实现。它不需要复现每一项、每个系数或每步 reward 计算，也不应机械地逐项优化 reward 公式。请用它们理解任务优先级和权衡。Driving prior 应协助车辆在安全且不过大的车距内跟随，同时遵守限速和信号灯；energy prior 应协助管理 FCS 功率，降低氢耗和部件退化，并使 SOC 接近目标。两个 agent 的任务和 reward 要保持区分。

Driving reward 是 following、comfort、traffic-light、speeding、绿灯越线一次性奖励、限速入口 shaping、静止时负加速度惩罚、达到限速时仍请求正加速度惩罚、terminal penalty 的总和。Energy reward 是 hydrogen、FCS 退化、电池 SOH 损失、EDU 退化和 SOC 各项之和。

**跟车项。** 使用 action 执行后的 ego 速度 `v_(t+1)`（m/s）和车距 `d_(t+1)`（m）：

- `D_safe = 2 + 0.5v + 0.0625v²`；`D_max = 10 + v + 0.0825v²`；`d_ref = 200 m`。
- `0 ≤ d < D_safe` 时：`c_follow = C₀ − C₁(1 − d/D_safe)³`，其中 `C₀=5`、`C₁=500`。
- `D_safe ≤ d ≤ D_max` 时：`c_follow = C₀`。
- `D_max < d < d_ref` 时：`c_follow = C₀ − C₂(d−D_max)/(d_ref−D_max)`，其中 `C₂=5`。
- `d ≥ d_ref` 时：`c_follow = C₃ × (本 step ego 前进距离 / 当前区域限速)`，其中 `C₃=5`。前进距离单位为 m，限速单位为 m/s，因此比值单位为秒。Lead 车距无效时这一项为 0。
- 最终跟车 reward 为 `β₀ c_follow`。

**舒适性。** `c_comfort = −|a_exec,t−a_exec,t−1| / 2.2`，使用实际执行加速度（m/s²）。Reward contribution 为 `β₅ c_comfort`。除数 2.2 来自请求加速度范围 `0.7−(−1.5)`。

**能耗与退化。** 实现按每步增量惩罚：氢耗项 = `−β₁ × 本 step 氢耗（g）`；FCS 项 = `−β₂ × 100 × FCS 退化增量`；电池项 = `−β₃ × 100 × 电池 SOH 损失增量`；EDU 项 = `−β₄ × 100 × EDU CLR 增量`。系数 100 用于将比例增量转为百分点。SOC 项为 `−β₆ × |SOC−0.5|`。

**红绿灯。** 每一步都根据 action 前的车速、信号灯距离、当前相位和相位剩余时间计算 reward；计算不局限于停止线前 100 m 的限速区。估算到停止线时间 `t₁ = 距离/当前速度`，并与当前相位剩余时间 `t₂` 比较。仅当车速 > 0 且 `t₁ < t₂` 时激活 reward。当前相位为绿灯时 raw reward 为 `+C_green`；为黄灯或红灯时为 `−C_non-green`。当前 `C_green=C_non-green=5`，再乘 `β₇=1`。若 `t₁ ≥ t₂`，当步不激活红绿灯 reward，在下一 step 重新计算。绿灯越过停止线另有一次性 +50 奖励。红灯越线会终止 episode；黄灯越线目前不会触发该终止条件。该 reward 使用当前观测相位；rule 决策还应根据给定有限预测窗口判断越线时的相位。

**限速。** 当前道路限速为 30 或 60 km/h。连续超速项使用 action 执行后的速度：`−100 × max(v_(t+1)−v_limit,0)/v_limit`。车辆 70 km/h 硬性最高速度与道路限速分开。进入较低限速区域前，环境使用 0.75 m/s² 的假设制动减速度预测入区速度：`v_entry = sqrt(max(0, v²−2×0.75×d_entry))`；`e = max(0,(v_entry−v_next_limit)/v_next_limit)`。入口 shaping 为 `β₈×(e_t−e_(t+1))`，其中 `β₈=500`。该项只在接近更低限速区域时计算；红灯不视为 0 限速。限速区域上下文切换或碰撞终止 step 会抑制可能产生的虚假正向边界奖励。

静止时负加速度的额外惩罚、达到限速后仍请求正加速度的额外惩罚，以及所有异常终止的额外 penalty 当前均为 0。

### 6. Episode 终止条件

- SUMO 报告碰撞或前向车距 ≤ 0（车辆重叠）时，episode 提前终止；当前没有额外 terminal penalty。
- Ego 越过停止线的瞬间若信号灯为红灯，仿真提前终止；黄灯越线不触发该终止条件，当前也没有额外 penalty。
- Ego 速度达到或超过当前道路限速的 120% 时，episode 提前终止；当前没有额外 penalty。30 km/h 限速区的阈值为 36 km/h，普通 60 km/h 道路的阈值为 72 km/h。70 km/h 车辆硬限速使普通道路上的 72 km/h 终止阈值无法达到。达到终止阈值前仍使用连续超速 reward。
- Lead 到达路线终点停车并静止，或 ego 完成路线时，episode 正常结束。
- 6000 s 时间上限会产生 truncation。
- 300 m 跟车距离诊断阈值和 1000 m 监控阈值只用于日志，不终止 episode。没有长时间静止终止条件。
- 碰撞、闯红灯和超速终止目前没有额外负的 terminal reward；主要影响是 episode 结束。

### 7. SAC 高斯 Rule Prior 与融合

Rule program 为每个 actor 输出其 `tanh` 前 latent 空间中的分布 prior，不返回物理加速度、kW 或采样后的 action。对 agent `i`（driving 或 energy），SAC actor 预测 `μ_actor^i` 和 `σ_actor^i = exp(log_std_actor^i)`。Rule 输出 `μ_prior^i` 和非负的 `σ_prior^i`。集成 wrapper 独立融合 mean 和 standard deviation：

`μ_f^i = (1−λ_μ^i) μ_actor^i + λ_μ^i μ_prior^i`

`σ_f^i = (1−λ_σ^i) σ_actor^i + λ_σ^i σ_prior^i`

Mean 与标准差的融合权重由外部配置提供，二者彼此独立，也与 rule program 无关；Driving 和 Energy 还可以分别配置。只有在没有其他指定时，才可将 0.5 作为示意默认值；不要将融合权重硬编码在 rule 里。对某个 agent/state，`active=False` 表示绕过该 prior，不做融合。若 `σ_prior=0` 且 `λ_σ<1`，由于 SAC actor 仍贡献一部分标准差，融合后的策略依然是随机策略。

SAC 随机选取 action 时，从融合分布采样 `z = μ_f + σ_f ε`，其中 `ε ~ Normal(0,1)`；之后执行 `u=tanh(z)` 并使用第 4 节中的物理 action 映射。确定性 action 使用 `u=tanh(μ_f)`。Rule 改变 actor 的分布，但不直接覆盖采样或映射后的 action。每个 1 s 决策 step 都根据当步 observation 计算 rule。

### 8. Rule Program 输入/输出接口

设计一个暴露 `predict(observation)` 的 Python 模块。程序接收完整原始 observation，形式为 NumPy 数组 `(15,)` 或 `(batch,15)`，而不是 SAC actor 的分离后归一化输入。将输入转换为 `float32`，检查形状和有限值，支持单条及 batch 输入，并确定性地返回两个 agent 的 prior。每行每个 agent 输出一个标量，建议使用 `(batch,1)`：

```python
{
    "driving": {"mean": driving_prior_mean, "std": driving_prior_std, "active": driving_active},
    "energy":  {"mean": energy_prior_mean,  "std": energy_prior_std,  "active": energy_active},
}
```

- 每个 actor 的 action 维度为 1。`mean` 必须为有限值，单位是 `tanh` 前的 latent Gaussian 空间。`std` 必须为有限的非负标准差，不是 `log_std` 或方差。
- `active` 是可选布尔值或 batch 布尔 mask；省略时默认 `True`。若 rule 对某个 agent/state 没有合理建议，可以设置 `active=False`。Rule program 仍须每个环境 step 调用一次。
- 一个 program 可为 driving、energy 或二者提供 prior。不得让某个 agent 的决策依赖其 observation 中不可见的另一个 agent 当前 action。
- 同一 observation 必须确定性地返回相同分布参数。不要在 `predict` 内采样 action 或添加探索噪声。
- 可以使用 NumPy。不要调用网络服务、API key、在线 LLM、文件 IO、环境全局变量或 SAC 内部对象。Program 只返回 prior 参数；融合、SAC 采样和物理 action 映射按上文定义执行。
- 新增的阈值或系数必须使用清晰命名的 rule 超参数并解释。不得暗中修改仿真环境、reward、动力系统参数或硬约束。明确列出不可直接观测的物理信号；不要编造数据。

### 9. 最小接口示例：超过当前限速时建议制动

此示例用于说明 prior 接口和 latent mean 定义；它不是完整驾驶策略，也不处理跟车、红绿灯或能量管理。目标归一化 action 为 `−0.999`，对应 prior mean `atanh(−0.999)≈−3.8002`，prior 标准差为 0.05。当 ego 速度没有超过当前限速时，驾驶 prior 不激活。Energy prior 始终不激活。

```python
import numpy as np

TARGET_U = -0.999
TARGET_MEAN = float(np.arctanh(TARGET_U))

def predict(observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != 15:
        raise ValueError("observation must have shape (15,) or (batch, 15)")
    if not np.all(np.isfinite(x)):
        raise ValueError("observation must contain only finite values")

    batch = x.shape[0]
    speed = x[:, 5]
    current_limit = x[:, 12]
    active_drive = (current_limit > 0.0) & (speed > current_limit + 1e-6)
    drive_mean = np.zeros((batch, 1), dtype=np.float32)
    drive_mean[active_drive, 0] = TARGET_MEAN
    drive_std = np.full((batch, 1), 0.05, dtype=np.float32)
    zeros = np.zeros((batch, 1), dtype=np.float32)
    inactive = np.zeros((batch, 1), dtype=bool)
    return {
        "driving": {
            "mean": drive_mean,
            "std": drive_std,
            "active": active_drive.reshape(batch, 1),
        },
        "energy": {"mean": zeros.copy(), "std": zeros.copy(), "active": inactive},
    }
```

例如，若 `observation[5]=18 m/s` 且 `observation[12]=16.6667 m/s`，Driving prior 激活并给出约 −3.8002 的 latent mean；Energy prior 不激活。最终随机 action 仍取决于外部融合权重和 SAC actor 分布。

### 10. 输出要求

请输出：(A) 完整 Python 模块及建议文件名；(B) 对 rule 逻辑、使用的 observation、触发条件、`active` mask、mean/std 数值范围和边界的准确简要说明。说明高斯 prior 如何经过外部融合和 `tanh` 影响 SAC actor，并区分 prior 与最终执行 action。明确列出假设、不可用的信号和新增的 rule 超参数。不要提供测试套件，不要要求执行代码，也不要声称代码已在 SUMO 中运行或验证。
