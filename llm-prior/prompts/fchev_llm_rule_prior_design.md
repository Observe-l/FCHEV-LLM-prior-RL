你是熟悉强化学习、SAC/HASAC、车辆纵向控制和燃料电池混合动力汽车（FCHEV）能量管理的工程师。请根据下述项目接口，为 FCHEV 环境设计一个可执行的 Python rule prior program。该程序会与学习得到的 HASAC actor 高斯策略融合；它不是独立控制器，也不能绕开环境物理约束。

### 1. 项目背景与优化目标

这是一个燃料电池混合动力电动汽车（FCHEV）驾驶策略项目。SUMO 提供道路、前车、信号灯和车辆运动仿真；Python 环境把车辆纵向动力学与燃料电池系统（FCS）、电池和电驱单元（EDU）模型连接起来；HASAC 使用两个独立 actor：driving actor 负责车辆加速度，energy actor 负责 FCS 请求功率。两个 actor 使用集中式 critic，并在 HASAC 中共同优化同一个团队 reward。

目标是在完成跟车和交通任务的同时：

- 正确跟随引导车，避免追尾，并尽量平顺地加速、减速；
- 遵守道路和路口限速，理解信号灯并安全通过路口；
- 减少氢耗、FCS 退化、电池 SOH 损失和 EDU 损耗；
- 维持电池 SOC 接近目标值 0.5。

控制周期为每步 1 s。每个 step 中，两个 actor 都根据该时刻的 observation 各自重新决策；rule program 也使用同一时刻的 observation 评估一次并生成相应 prior。该步的 action 和 prior 只用于当前 step，并在下一个 step 根据新 observation 重新计算。

SOC 偏差是能量管理目标，不是电池退化量。现有观测没有直接包含电机转矩、即时 EDU 损耗、FCS 功率请求、电池功率或氢耗；规则不得假设这些量能从 observation 直接读取。如果需要根据它们决策，只能使用明确标注的近似估计，并说明该估计的限制。

### 2. 当前仿真环境与精确数值

**道路与任务：**

- 路网中有 25 个信号灯路口，相邻路口间距 1000 m。
- 一个 episode 最长 6000 s；每个 RL step 为 1 s。
- ego 和引导车均从 0 m/s 起步，初始车距 10 m。引导车目标速度为 13 m/s（46.8 km/h）。
- 引导车采用 SUMO IDM 跟车模型；其 vType 参数为 accel=0.7 m/s²、decel=1.5 m/s²、emergencyDecel=1.5 m/s²、minGap=3 m、tau=1.5 s、sigma=0、speedDev=0。引导车的常规和紧急最大减速度都与 ego 的动作减速度下限 1.5 m/s² 一致。ego 的 SUMO vType 仍保留 emergencyDecel=6.68 m/s²；ego 的常规动作映射范围为 [-1.5, 0.7] m/s²。引导车在距离信号灯 200 m 内及路口内部受 30 km/h 速度上限约束。
- 25 个信号灯之间共有 24 段相邻路段，每段给引导车设置一个随机停车点；每次停车 50–100 s，停车点距该段两端至少 100 m。随机场景由 seed 决定，同一 seed 下信号初相位和停车计划可复现。
- 信号灯周期 90 s：绿灯 40 s、黄灯 3 s、红灯 47 s。各信号灯初始相位不同，因此不会全部同步。
- 普通道路限速 60 km/h（16.6667 m/s）；启用的路口限速区为 30 km/h（8.3333 m/s），长度为停止线前 100 m，并覆盖路口内部。驶出路口后恢复 60 km/h。
- Observation 中的限速与信号灯状态分开表示：红灯不会把限速 observation 变成 0。进入 30 km/h 区域后，下一个限速区域是路口后的 60 km/h 区域。
- 车辆最高速度为 70 km/h（19.4444 m/s）。它是硬上限；当请求加速会使速度超过上限时，环境会截断加速度。
- 引导车在终点停车；当引导车到达终点停车位置并基本停稳，episode 正常结束。车辆完成路线也会结束 episode。

**车辆与动作边界：**

- 车辆质量 7000 kg，长度 8 m，车轮半径 0.4 m，迎风面积 5.8 m²，风阻系数 0.55，滚阻系数 0.0085，旋转质量系数 1.05，空气密度 1.225 kg/m³，重力加速度 9.8 m/s²，道路坡度 0°。
- actor 的 driving action 经 tanh 后为 u_d ∈ [-1, 1]，再映射成请求加速度：u_d < 0 时 a_req = 1.5 u_d（[-1,0] 对应 [-1.5,0] m/s²）；u_d ≥ 0 时 a_req = 0.7 u_d（[0,1] 对应 [0,0.7] m/s²）。
- 环境禁止倒车：车辆静止时，负加速度请求会被物理处理截断，不会让车辆向后行驶。
- ego 的 SUMO speed mode 设为 0，由环境直接设置速度，因此 SUMO 不会替 ego 执行跟车、限速或红灯停车监督；道路限速仍通过 reward/终止条件处理，70 km/h 是单独的车辆硬上限。
- actor 的 energy action 经 tanh 后为 u_e ∈ [-1,1]，再映射为 FCS 请求功率 P_req = 25(u_e+1) kW，即 [0,50] kW。
- FCS 最大功率为 50 kW，DC/DC 效率为 0.95。FCS 实际执行功率可能受能量分配、EDU 需求和电池物理限值影响，不保证等于请求值。
- 电池为 A123 26650 LiFePO₄ 电芯模型，配置为 106 串、134 并；单体标称电压 3.3 V、容量 2.3 Ah。SOC 初值和目标值均为 0.5，SOC 工作范围 [0.15, 0.9]；放电上限 2C、充电上限 1C，单体电压范围 2.0–3.6 V，初始 core/surface/coolant 温度均为 25°C。电池采用二阶 RC 电热模型和数据查表。
- EDU 使用 EPA Generic IPM 150 kW EDU 数据集的整套 EDU map，包含 motor、inverter、gearbox 的组合边界，不是 motor-only map；传动比为 9.5。数据中未提供额定效率，不要把峰值效率称为额定效率。环境使用总 EDU 效率/损耗 map 和有效工作区，不应再重复添加 inverter 或 gearbox 损耗。
- FCS 氢耗表来自 Lin and Zheng (2011) 的原始 50 kW FCS 多项式数据，未缩放为其他额定功率；老化模型同时计算启动、低功率、高功率和负载变化引起的退化。EDU 老化参考点为机械功率 75 kW、转矩 156 Nm、转速 4591 rpm，参考寿命 30000 h；电池按当前电热与老化模型更新 SOH。
- 没有跟车监督器或信号灯监督器替 RL 控车。SUMO 不会替 ego 自动跟车或自动等红灯；但电池/FCS/EDU 可行域、禁止倒车和最高车速硬限制仍然生效。请求动作可能因这些物理约束而与实际执行动作不同。

### 3. Observation、actor 输入与量纲

rule program 的输入是**完整、原始、未归一化的 15 维 observation**，顺序不可改变。环境会把没有下一个信号灯时的信号状态编码为绿灯 1、距离 30000 m、剩余时间 0 s；30000 m 是“没有可见目标/超出有效范围”的哨兵值。

| 索引 | 名称 | 单位 / 编码 |
|---:|---|---|
| 0 | 电池 SOC | fraction，通常 [0.15, 0.9] |
| 1 | 电池 SOH | fraction，初值 1 |
| 2 | FCS SOH | fraction，初值 1 |
| 3 | EDU SOH | fraction，初值 1 |
| 4 | ego 当前加速度 | m/s² |
| 5 | ego 当前速度 | m/s |
| 6 | 引导车当前加速度 | m/s² |
| 7 | 引导车当前速度 | m/s |
| 8 | ego 到前车的距离 | m，最大观测值 30000 |
| 9 | 下一个信号灯状态 | 红灯 0、绿灯 1、黄灯 2 |
| 10 | 到下一个信号灯停止线的距离 | m，最大 30000；无信号灯时为 30000 |
| 11 | 当前信号相位剩余时间 | s，通常 [0, 100]；无信号灯时为 0 |
| 12 | 当前区域限速 | m/s；主路 16.6667、路口区 8.3333 |
| 13 | 下一个限速区域的限速 | m/s |
| 14 | 到下一个限速区域入口的距离 | m，最大 30000 |

HASAC 内部 driving actor 接收原始 observation 的索引 4–14 共 11 维；仅将信号灯类别除以 2，其他值仍用上述 SI 单位。Energy actor 接收 5 维：[当前 ego 加速度、SOC、电池 SOH、FCS SOH、EDU SOH]。但 rule program 统一接收完整的原始 15 维 observation，而不是这两个 actor 的拆分输入。

### 4. Action 的物理映射与实际执行

每个 actor 输出一个标量高斯策略，经 tanh 得到归一化 action：

- driving actor：u_d = tanh(z_d)，然后按上节的分段线性关系映射成请求加速度 [-1.5, 0.7] m/s²；
- energy actor：u_e = tanh(z_e)，映射成 0–50 kW FCS 请求功率。

这两个 action 是两个独立 actor 的输出，不存在固定的一一对应关系。环境随后执行车辆动力学和功率分配可行性检查；最大车速、禁止倒车、电池功率/SOC/电流/电压边界、EDU torque-speed/效率有效区及 FCS 限值都可能截断或改变请求。Rule prior 不能绕过或修改这些限制。

### 5. 当前 reward 定义

HASAC 的 driving reward 与 energy reward 会相加，作为两个 actor **共同收到的同一个团队 reward**。以下系数来自当前配置：

| 系数 | 当前值 | 主要作用 |
|---|---:|---|
| β₀ | 1 | 跟车 |
| β₁ | 0.2 | 氢耗 |
| β₂ | 5000 | FCS 退化增量 |
| β₃ | 100000 | 电池 SOH 损失 |
| β₄ | 100000 | EDU 退化增量 |
| β₅ | 2 | 驾驶舒适性 |
| β₆ | 10 | SOC 偏差 |
| β₇ | 1 | 信号灯预测奖励 |
| β₈ | 500 | 进入较低限速区的速度 shaping |

主要分量如下。rule 可辅助改善这些目标，但不能假定某个单项 reward 与总 reward 等价：

**跟车项。** ego 速度 v 使用 m/s，车距 d 使用 m：

- D_safe = 2 + 0.5v + 0.0625v²；D_max = 10 + v + 0.0825v²；d_ref = 200 m。
- 0 ≤ d < D_safe：c_follow = C₀ − C₁(1 − d/D_safe)³，其中 C₀=5、C₁=500。
- D_safe ≤ d ≤ D_max：c_follow = C₀。
- D_max < d < d_ref：c_follow = C₀ − C₂(d−D_max)/(d_ref−D_max)，其中 C₂=5。
- d ≥ d_ref：c_follow = C₃ × (本 step ego 前进距离 / 当前区域限速 m/s)，其中 C₃=5。前进距离单位为 m，因此比值单位为秒。没有有效前车距离时该项为 0。
- 最终跟车 reward 为 β₀ c_follow。

**舒适性。** c_comfort = −|a_t−a_(t−1)| / 2.2，单位加速度变化以 m/s² 计；reward 为 β₅ c_comfort。2.2 来自请求加速度范围 0.7−(−1.5)。

**能量与退化。** 当前实现按每个 step 的增量惩罚：氢耗项为 −β₁ × 本 step 氢耗(g)；FCS 项为 −β₂ × 100 × FCS 退化增量；电池项为 −β₃ × 100 × 电池 SOH 损失增量；EDU 项为 −β₄ × 100 × EDU CLR 增量。100 把 fraction 增量换成百分数点。SOC 项为 −β₆ × |SOC−0.5|。

**信号灯。** 根据当前速度估算到停止线的时间 t₁ = 距离/速度。仅在速度大于 0 且 t₁ 小于当前相位剩余时间 t₂ 时计分：当前相位为绿灯，raw reward 为 +C_green；当前相位为黄灯或红灯，raw reward 为 −C_non-green。当前 C_green=C_non-green=5，乘 β₇=1。绿灯越过停止线另有一次性 +50 bonus。闯红灯仍会结束 episode；黄灯本身不会触发该红灯终止条件。

**限速。** 当前限速为 30 或 60 km/h，连续超速项为 −100 × max(v−v_limit,0)/v_limit。车辆 70 km/h 硬上限独立于 posted limit。进入较低限速区时，使用减速度估计值 0.75 m/s² 预测入口速度：v_entry = sqrt(max(0, v²−2×0.75×d_entry))，e = max(0,(v_entry−v_next_limit)/v_next_limit)，entry shaping 为 β₈×(e_t−e_(t+1))，β₈=500。它只针对接近较低限速区，不把红灯当作零限速；代码会在限速区上下文切换或碰撞终止步抑制不合理的正向边界 bonus。

当前零速负加速度额外惩罚、达到限速后继续请求正加速度的额外惩罚以及异常终止额外惩罚均为 0。

### 6. Episode 终止条件

- 碰撞（SUMO 报告碰撞）或前车间距 ≤ 0（车辆重叠）会提前终止；当前额外 terminal penalty 为 0。
- ego 越过停止线时，如果越线前信号灯为红灯，会提前终止；当前额外 penalty 为 0。黄灯越线不由这一条件终止。
- 当 ego 速度达到或超过当前有效限速的 120% 时提前终止；当前额外 penalty 为 0。30 km/h 区域阈值为 36 km/h；普通 60 km/h 道路阈值为 72 km/h，但车辆 70 km/h 硬上限使普通道路上无法达到该终止阈值。低于终止阈值时，连续超速 reward 仍然生效。
- 引导车到达终点停车点并停稳或 ego 路线完成时，episode 正常结束。
- 6000 s 时间上限会 truncation。
- 车距 300 m 的诊断阈值与 1000 m 的监测阈值只用于记录，不会触发终止；没有“长时间不动”的终止条件。
- 碰撞、信号灯或超速导致的提前终止目前不额外施加负的 terminal reward；主要效果是 episode 结束。
- 当前 HASAC 配置的折扣因子 gamma=0.99；训练期间采用随机 warm-up 与高斯策略采样，Rule prior 只参与 actor 策略阶段。
- 环境区分 terminated 与 time-limit truncated；当前训练代码把 terminated 或 truncated 都作为 replay 中的 done 标志。

### 7. Rule prior 与 HASAC actor 的融合

rule program 不直接给出物理加速度或 kW，而是对每个 actor 输出其 tanh 之前的 latent Gaussian 分布参数：mean μ_prior 和标准差 σ_prior。HASAC 的 actor 网络分别预测 mean 和 log_std，log_std 截断在 [-20, 2]，再令 σ_actor=exp(log_std)。两个参数独立融合：

- μ_f = (1−λ₀) μ_actor + λ₀ μ_prior
- σ_f = (1−λ₁) σ_actor + λ₁ σ_prior

当前 λ₀=0.5、λ₁=0.5。融合后训练时采样 z=μ_f+σ_f ε，ε~Normal(0,1)；确定性评估使用 z=μ_f。接着 u=tanh(z)，最后按第 4 节映射到物理 action。由此 rule 通过改变 actor 的 mean/std 影响动作，而不是覆盖 actor action。σ_prior=0 表示规则建议低不确定性，并不意味着融合后或采样后的最终 action 必然确定；只要 λ₁<1，actor 的标准差仍会保留一部分。

训练开始的前 1000 个环境 step 是 replay warm-up，动作由均匀随机采样产生，此阶段不会调用 actor 或融合 prior。之后的 actor 采样和 HASAC 更新才使用 prior。

### 8. Rule program 的输入/输出接口

生成的模块由外部配置 `llm_prior.program_name` 选择，对应文件路径为 `llm-prior/programs/<program_name>.py`；命令行也可用 `--prior-program-name <program_name>` 覆盖。prior 必须同时在配置中启用。程序必须实现：

    def predict(observation):
        ...

`observation` 是 NumPy 数组，形状为单条状态 `(15,)` 或批量状态 `(batch, 15)`，数值是第 3 节所示完整原始 observation。推荐程序将输入转换为 float32，内部统一转成 batch，再输出 `(batch,1)` 的数组。输出必须是下面的 mapping，并同时包含 driving 与 energy：

    {
        "driving": {"mean": ..., "std": ..., "active": ...},
        "energy":  {"mean": ..., "std": ..., "active": ...},
    }

- 两个 actor 的 action 维度均为 1，因此每个 mean/std 对应一个有限标量，推荐数组形状为 `(batch, 1)`。
- mean/std 必须是在 tanh 前的 latent Gaussian 空间；std 必须有限且 ≥0。
- active 是可选布尔 mask，形状推荐 `(batch,1)`；若不提供则默认为 True。active=False 表示该状态下完全不融合该 actor 的 prior。
- 同一个程序可以只影响 driving、只影响 energy，也可以同时影响两者。未打算控制的 actor 应设 active=False。
- 程序应支持单条和 batch 输入，拒绝错误维度、NaN、Inf；对相同输入应确定性返回相同分布参数。不要在 predict 中抽样动作。
- 可以使用 NumPy；不要依赖网络服务、API key、LLM 在线调用、文件读写、环境全局变量或 HASAC 内部代码。程序只生成分布参数，由现有 HASAC 完成融合、采样和物理映射。
- 规则里新增的阈值/系数必须作为清晰命名的超参数并解释依据；不要悄悄改动仿真环境、reward、动力系统参数或硬约束。不可获得的物理信号要明确说明，不得编造。

### 9. 简单接口示例：超过当前限速时建议减速

下面是一个最小示例，用于展示接口和 latent mean 的定义；它不是完整驾驶策略，也不处理跟车、信号灯或能量管理。目标归一化动作设为 −0.999，其 pre-tanh mean 为 atanh(−0.999)≈−3.8002。当前速度不超限时，prior inactive；超限时只激活 driving prior。energy prior 始终 inactive。

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
        zeros = np.zeros((batch, 1), dtype=np.float32)
        inactive = np.zeros((batch, 1), dtype=bool)
        return {
            "driving": {
                "mean": drive_mean,
                "std": zeros.copy(),
                "active": active_drive.reshape(batch, 1),
            },
            "energy": {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "active": inactive,
            },
        }

例如，若 observation[5]=18 m/s、observation[12]=16.6667 m/s，则 driving prior 的 mean≈−3.8002、std=0、active=True。以 λ₀=0.5 为例，融合后 mean 是 actor mean 与 −3.8002 的平均值；之后仍会执行高斯采样（训练时）、tanh 和物理 action 映射。环境仍可能根据最高速度、禁倒车和动力系统可行域截断结果。

### 10. 你的输出要求

请只输出下面两部分，内容完整到可以审查并直接集成：

**A. 可执行 Python program**

提供完整源代码，建议文件名 `<program_name>.py`，放入 `llm-prior/programs/` 后可由现有 HASAC loader 导入。必须实现上面的 `predict(observation)` 接口；若使用参数化 rule，也可提供 `build_prior(config)`，但 `predict` 能力必须存在。代码应包括输入形状/有限值检查、单条与 batch 处理、两个 agent 的输出，以及必要的注释。不要输出伪代码、省略号或依赖未提供文件的片段。

**B. Rule 效果说明**

说明 rule 的目标、使用了哪些 observation 索引、每条规则的触发条件与优先级、两个 actor 的 mean/std/active 如何设置、λ 融合后如何影响动作、边界和缺失目标如何处理、用了哪些新增超参数以及理由。给出至少一个 15 维 observation 的测试例子和对应输出摘要，并说明如何把程序名填入 `llm_prior.program_name`、如何做接口 smoke test。诚实说明 rule 能改善什么、不能保证什么，尤其不能保证未观测的电机损耗最优或覆盖 HASAC 学到的策略。

不得宣称示例程序已经在 SUMO 中训练或验证。不得修改环境参数，不得给 observation 增加字段，不得返回物理 action，也不得让程序绕过 episode 终止条件或车辆/电池/EDU 物理约束。
