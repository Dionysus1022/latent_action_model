# LeWM + Diffusion 技术路线（讨论稿）
lewm github仓库：https://github.com/lucas-maes/le-wm  
diffusion 规划仓库：https://github.com/hustvl/diffusiondrive
lewm论文原文：https://arxiv.org/abs/2603.19312
## 1. 问题背景

我们当前使用的 LeWM（Large Efficient World Models）已经具备较强的世界模型能力：

- 可以把当前观测编码到 latent 空间
- 可以通过 latent dynamics predictor 预测未来状态

但 LeWM 当前最大的实际瓶颈不在 world model 本身，而在**测试时的规划方式**：

- 当前规划采用 **MPC / CEM**
- 每次决策都要在线采样大量候选动作序列
- 再用 world model rollout 和代价函数筛选最优动作块
- 因此评估和部署速度都比较慢

也就是说，LeWM 现在属于：

$$
\text{World Model} + \text{Test-time Planning}
$$

而不是：

$$
\text{World Model} + \text{Direct Planner}
$$

我们的目标是把在线规划的计算负担尽量转移到训练阶段，让测试时只需一次轻量前向或极少步生成即可输出动作序列。

---

## 2. 核心目标

我们准备做的是：

> 在保留 LeWM 现有 world model（encoder + predictor）的基础上，增加一个可学习的快速 planner，用来替代当前慢速的在线 CEM/MPC。

最终希望得到：

$$
(z_t, z_g) \rightarrow \hat{u}
$$

其中：

- \(z_t\)：当前 latent 状态
- \(z_g\)：目标 latent 状态
- \(\hat{u}\)：一段短时程动作块（action chunk）

也就是说，测试时不再显式运行 CEM，而是由一个训练好的 planner 直接输出动作块。

---

## 3. 为什么考虑 DiffusionDrive 这条路线

我们不打算直接走 Dreamer / HIQL 这类大改结构的路线，原因是：

1. **LeWM 已经有 world model**，当前最痛的点不是表征学习，而是规划太慢。
2. Dreamer / HIQL 都需要额外引入：
   - value / critic
   - actor / policy learning
   - 更复杂的 RL 训练范式  
   这会把问题从“加速规划”变成“重做决策层”。
3. DiffusionDrive 代表的是另一条更贴合当前问题的路线：

> 把测试时的规划过程，蒸馏成训练好的生成式 planner。

所以我们的思路不是“照搬自动驾驶模型”，而是：

> 借鉴 DiffusionDrive 的思想，把 LeWM 当前的 CEM planner 蒸馏成一个目标条件的生成式动作规划头。

---

## 4. 总体技术路线

我们准备采用的路线可以概括为：

$$
\text{LeWM encoder/predictor} + \text{Teacher CEM planner} + \text{Diffusion planner head}
$$

具体分成两部分。

### 4.1 保留的部分

继续使用 LeWM 现有能力：

- encoder：把 observation 编码到 latent
- predictor：在 latent 空间中 rollout
- 现有 CEM planner：作为 teacher，用于离线产生监督信号

### 4.2 新增的部分

新增一个 **goal-conditioned planner head**：

- 输入：当前 latent + 目标 latent
- 输出：一段动作块
- 训练：模仿 teacher planner
- 推理：一次前向或极少步 diffusion 生成

---

## 5. 问题形式化

设：

- 单步动作维度为 \(d_a\)
- 规划 horizon 为 \(H\)

则一个动作块可以写成：

$$
u = [a_t, a_{t+1}, \dots, a_{t+H-1}] \in \mathbb{R}^{H d_a}
$$

注意：

- 这里**不要预先写死动作维度**
- 应以真实环境接口为准

例如：

- 如果当前环境单步动作是 2 维，且 \(H=5\)，则动作块维度是 \(10\)
- 如果后续换成高层动作抽象，维度再相应调整

因此统一记为：

$$
D = H \cdot d_a
$$

目标就是学习：

$$
f_\phi(z_t, z_g) \rightarrow u
$$

---

## 6. Teacher 数据构建

为了训练 fast planner，我们先使用当前 LeWM + CEM 作为 teacher。

对每一个可规划时刻，构造样本：

$$
(z_t, z_g, u^*, \{u^{(m)}, J^{(m)}\}_{m=1}^{M})
$$

其中：

- $z_t$：当前 latent
- $z_g$：目标 latent
- $u^*$：teacher 选出的最优动作块
- $u^{(m)}$：top-M elite 动作块
- $J^{(m)}$：对应的规划 cost

这里 teacher dataset 的核心思想是：

> 让当前慢速但高质量的规划器，给后续快速规划器提供监督数据。

这一步相当于把 test-time planning 变成 train-time supervision。

---

## 7. 最终想做的模型：Anchor-Conditioned Diffusion Planner

我们的最终版本不是直接从零回归动作块，而是采用“**动作锚点 + 截断 diffusion**”的设计。

### 7.1 动作锚点（Action Anchors）

先把 teacher planner 输出的优质动作块集合拿出来，只在 **train split** 上做 K-means 聚类，得到 \(K\) 个动作锚点：

$$
\mathcal{A} = \{c_1, c_2, \dots, c_K\}, \qquad c_k \in \mathbb{R}^{D}
$$

这些锚点表示“几种典型的短时程动作模式”。

注意：

- 聚类对象是 **teacher 动作块**
- 不是原始环境动作
- 也不是 validation / test 数据

这样做的作用是：

> 让 planner 不是从完全随机开始生成，而是从几个合理的动作模式附近出发。

### 7.2 条件输入

planner 的条件输入定义为：

$$
x = [z_t,\ z_g,\ z_g - z_t]
$$

然后经过一个条件编码器：

$$
h = \phi(x)
$$

其中 \(h\) 是 planner 的条件特征。

直觉上：

- \(z_t\)：告诉模型“我现在在哪”
- \(z_g\)：告诉模型“我要去哪”
- \(z_g-z_t\)：告诉模型“当前和目标之间差多少”

### 7.3 扩散输入构造

对于每个动作锚点 \(c_k\)，构造带噪动作块：

$$
\tilde{u}_i^{\,k} = \sqrt{\bar{\alpha}_i} c_k + \sqrt{1-\bar{\alpha}_i}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0,I)
$$

其中：

- \(i\) 是 diffusion timestep
- \(T_{\text{trunc}}\) 很小，只取很短的截断扩散区间
- 推理时只做极少步数，比如 2 到 4 步

这一步不是传统“大扩散”，而是：

> 围绕动作锚点做小范围生成和修正。

### 7.4 Planner 网络输出

在每个去噪步骤，网络输出：

$$
\{(\hat{u}_k,\hat{s}_k)\}_{k=1}^{K}
$$

其中：

- \(\hat{u}_k\)：第 \(k\) 个候选动作块
- \(\hat{s}_k\)：第 \(k\) 个候选动作块的分数

也就是说，planner 一次会给出多条候选动作块，并同时给它们评分。

最终选择：

$$
k^* = \arg\max_k \hat{s}_k,\qquad \hat{u} = \hat{u}_{k^*}
$$

---

## 8. 训练目标

训练时我们用 teacher planner 的输出监督 diffusion planner。

### 8.1 正锚点定义

先找最接近 teacher 最优动作块的锚点：

$$
k^+ = \arg\min_k \|u^* - c_k\|_2^2
$$

### 8.2 分类损失

让模型学会选对动作模式：

$$
L_{\text{cls}} = \mathrm{CE}(\hat{s}, k^+)
$$

### 8.3 重建 / 去噪损失

对正锚点对应的候选动作块进行监督：

$$
L_{\text{rec}} = \|\hat{u}_{k^+} - u^*\|_1
$$

如果最终实现采用“预测噪声”的 diffusion 形式，也可以改成噪声预测损失；但第一版 diffusion 建议优先采用更容易调试的直接动作块重建方式。

### 8.4 目标一致性损失（LeWM 特有优势）

利用现有 world model predictor，对预测动作块 rollout：

$$
\hat{z}_{t+H} = F(z_t,\hat{u}_{k^+})
$$

然后要求它靠近目标 latent：

$$
L_{\text{goal}} = \|\hat{z}_{t+H} - z_g\|_2^2
$$

### 8.5 总损失

总损失定义为：

$$
L = L_{\text{cls}} + \lambda_{\text{rec}} L_{\text{rec}} + \lambda_{\text{goal}} L_{\text{goal}}
$$

---

## 9. 推理流程

最终推理时不再运行 CEM，而是：

1. 编码当前 observation，得到 \(z_t\)
2. 编码目标 observation，得到 \(z_g\)
3. 构造条件特征 \(h\)
4. 从动作锚点附近初始化若干带噪候选动作块
5. 做极少步数的去噪
6. 输出 \(K\) 个候选动作块及其分数
7. 选择最高分动作块执行

即：

$$
(z_t, z_g) \rightarrow \text{Diffusion Planner} \rightarrow \hat{u}
$$

这样测试时就不再需要大量在线搜索。

---

## 10. 为什么这条路线适合 LeWM

这条路线的好处主要有三点。

### 10.1 改动小

不需要重写整个 LeWM。

保留：

- encoder
- predictor
- 现有 planner teacher

只是在顶层新增一个 planner head。

### 10.2 直接针对当前瓶颈

LeWM 当前最大问题是 **MPC/CEM 规划慢**。  
这个方案正面解决的是：

> 如何把慢的在线规划蒸馏成快的离线 planner。

### 10.3 比 HIQL 更贴题

HIQL / Dreamer 更适合：

- 长时程策略学习
- 分层决策
- 完整 RL 训练范式

而我们现在更需要的是：

- 保留现有 world model
- 替换当前慢速 planner
- 让 planning 从 online 变成 amortized

所以 diffusion planner 比 HIQL 更适合作为 LeWM 当前阶段的增强路线。

---

## 11. 实验计划

我们最终希望做三组比较。

### 11.1 与原始 LeWM + CEM 对比

比较：

- success rate
- 总评估时间
- 单次 planning 调用时间
- 目标 latent 误差

### 11.2 与简化版 fast planner 对比

先做一个更简单的 baseline，例如：

- 单峰直接回归 planner
- 或锚点 + 残差 + 打分版 planner（无 diffusion）

然后比较 diffusion 版本是否真的带来收益。

### 11.3 消融实验

主要做：

- anchor 数量 \(K\)
- diffusion 步数 \(T_{\text{trunc}}\)
- 是否加入 goal-alignment loss
- 是否使用 top-M teacher elites

---

## 12. 研发顺序（非常重要）

我们不会一步到位直接上完整 diffusion，而是按下面顺序推进。

### 阶段 1：最简单 baseline

先做一个单峰 planner：

$$
(z_t,z_g,z_g-z_t)\to\hat{u}
$$

验证“摊销规划”这条路本身是否可行。

### 阶段 2：多候选 planner

再做：

- 锚点
- 多候选
- 打分

验证多模态建模是否有收益。

### 阶段 3：最终 diffusion planner

最后再做：

- 动作锚点
- 截断 diffusion
- 条件去噪
- 评分头

也就是完整的 **LeWM + Diffusion** 版本。

---

## 13. 一句话总结

我们的核心思路是：

> 保留 LeWM 当前的 world model，把原本测试时依赖 CEM/MPC 的在线规划过程，蒸馏成一个目标条件的快速生成式 planner；最终采用“动作锚点 + 截断 diffusion + 候选打分”的方式，在保持规划质量的同时显著降低测试时延。

如果进一步压缩成一句话：

> **LeWM 负责理解和预测世界，Diffusion Planner 负责快速生成动作。**
