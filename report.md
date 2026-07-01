# Comparison Report

数据来源：`result.md` 的 `Final Summary`。实验覆盖 4 个数据集：`cube`、`pusht`、`reacher`、`tworoom`；比较 6 个方法：`mpc_cem`、`ours_full`、`lgbs_gcidm`、`lgbs_mppi`、`lgbs_icem`、`lgbs_gradient`。

主评价指标以成功率为第一优先级；成功率接近时，比较总评估时间、planner 总耗时、最终目标距离、成功步数和 action jerk。`Speedup vs MPC` 与 `Planning Speedup vs MPC` 均以同任务 `mpc_cem` 为 1.0。

## Inference Pipelines

本节按代码实现描述各方法在评估时的数据流。所有方法的外层评估协议一致：从 HDF5 数据集中采样 `num_eval` 个 episode 起点，按 `goal_offset_steps` 取目标观测，在环境中 closed-loop 执行 `eval_budget` 步，并记录成功率、耗时、planner 调用次数和 trajectory quality。

### Shared Evaluation Inputs

所有方法共享以下输入：

- Dataset：任务对应的 HDF5 数据集，例如 `cube_single_expert.h5`、`pusht_expert_train.h5`、`reacher.h5`、`tworoom.h5`。
- Episode start：由 `sample_eval_episode_starts` 或 LGBS 的 `sample_eval_episodes` 从数据集中采样。
- Current observation：环境当前返回的 `info_dict["pixels"]`，shape 通常为 `[B, history, H, W, C]`，预处理后变为 `[B, history, C, H, W]`。
- Goal observation：由数据集按 `goal_offset_steps` 取出的 `info_dict["goal"]`，同样会做 image transform。
- World model checkpoint：`swm.policy.AutoCostModel` 或 LGBS 中加载的 LeWM/JEPA 模型。
- Action/history normalizer：对非像素字段使用 `StandardScaler`，主要包括 `action` 和任务相关状态字段。
- PlanConfig：本实验统一使用 `horizon=5`、`receding_horizon=5`、`action_block=5`。因此一次 planner 输出的是 5 个 action block，每个 block 包含 5 个原始动作，总共对应 25 个环境 step。

### `ours_full`: Diffusion Planner + World-Model Rerank + Refinement

代码入口：`eval.py` 中 `planner_type == "diffusion"` 分支，运行时策略为 `diffusion/policy.py::DiffusionPlannerPolicy`。

输入：

- `policy`：LeWM world model checkpoint，用于编码当前/目标图像，并为候选 action chunk 打分。
- `diffusion_bundle`：训练好的 diffusion planner bundle，包含 action anchors、模型权重、扩散步数、action chunk 维度等。
- `prepared_info`：经 `BasePolicy._prepare_info` 处理后的当前观测、目标观测和 action history。
- 推理超参：`diffusion_num_candidates`、`diffusion_truncation_steps`、`diffusion_start_timestep`、`diffusion_eta`、`diffusion_noise_scale`、`diffusion_sampling_temperature`、`diffusion_selection_mode`、`diffusion_runtime_execute_steps`。
- refinement 超参：`enabled`、`steps`、`step_size`、`topk`、`goal_weight`、`prior_weight`、`smoothness_weight`、`grad_clip_norm`。

模块和数据流：

1. `eval.py` 加载 world model：
   `swm.policy.AutoCostModel(cfg.policy)`，放到 CUDA，设为 eval，并冻结参数。
2. `DiffusionPlannerPolicy.from_bundle(...)` 加载 diffusion planner：
   bundle 内的 anchors shape 为 `[K, action_chunk_dim]`，其中 `action_chunk_dim = action_chunk_horizon * action_dim`。
3. 每次需要 replan 时，`get_action` 调用 `plan_actions(prepared_info)`。
4. `encode_current_goal(prepared_info)`：
   - 输入：`pixels: [B, history, C, H, W]`，`goal: [B, history, C, H, W]`
   - world model 编码当前图像和目标图像
   - 输出：`z_cur: [B, latent_dim]`，`z_goal: [B, latent_dim]`
5. `generate_candidates(prepared_info)`：
   - diffusion model 以 `(z_cur, z_goal)` 为条件。
   - 先把 anchors 扩展为 `[B, K, action_chunk_dim]`。
   - 在 `start_timestep` 对 anchors 加噪，得到 `initial_noisy_candidates`。
   - 按 `truncation_timesteps` 做截断反向扩散，每一步调用 `DiffusionPlannerModel.forward(...)` 预测 `x0_pred/refined_actions` 和 `score_logits`。
   - 输出：
     - `candidates: [B, K_eff, action_chunk_dim]`
     - `score_logits: [B, K_eff]`
     - `initial_noisy_candidates`
     - `final_noisy_state`
6. `score_candidates_with_world_model(prepared_info, candidates)`：
   - 将 flat action chunk reshape 成：
     - `candidate_steps: [B, K, plan_horizon, action_dim]`
     - `candidate_blocks: [B, K, receding_horizon, action_block * action_dim]`
   - 将 `prepared_info` 扩展到 `[B, K, ...]`。
   - 用 world model rollout 得到每个候选的 predicted latent。
   - 用 world model criterion 计算 goal cost。
   - 输出：`world_model_costs: [B, K]`。
7. `refine_candidates_with_world_model(...)`，如果 refinement 开启：
   - 先按 `world_model_costs` 选择 `refinement_topk` 个候选；如果 `topk=None`，优化全部候选。
   - 对候选 action chunk 开启梯度，但 world model 参数冻结。
   - 用 `latent_rollout(...)` 从 `z_cur` 和 action blocks 预测 terminal latent。
   - 优化目标：
     - `goal_cost = MSE(z_terminal, z_goal)`
     - `prior_cost = MSE(candidate, initial_candidate)`
     - `smoothness_cost = mean squared adjacent action difference`
     - `total_cost = goal_weight * goal_cost + prior_weight * prior_cost + smoothness_weight * smoothness_cost`
   - 做 `refinement_steps` 步梯度下降：
     `candidate = candidate - step_size * grad`
   - 可选 `grad_clip_norm` 限制梯度范数。
   - 输出：替换了 top-k refined candidates 的完整候选集合。
8. refinement 后重新调用 `score_candidates_with_world_model(...)`，得到 refined 后的 `world_model_costs`。
9. `select_best_candidates(...)`：
   - `wm_only`：选择 world model cost 最低的候选。
   - `score_only`：选择 diffusion score 最大的候选。
   - `hybrid`：对 world-model cost 和 model score 做 row-wise normalization，用 `normalized_score - normalized_cost` 选择。
   - 当前主实验使用 `wm_only`。
   - 输出：`selected_candidates: [B, action_chunk_dim]`，`selected_indices: [B]`。
10. `unflatten_action_chunk(...)`：
    - 将 selected chunk 转成 `selected_plan: [B, plan_horizon, action_dim]`，其中 `plan_horizon=25`。
11. 可选 action residual corrector：
    - 如果 corrective correction 启用，会对 selected plan 做 learned residual 修正。
12. action buffer：
    - 只将前 `runtime_execute_steps` 个 action 放入 buffer。
    - 每次环境 step 从 buffer 弹出一个 `[B, action_dim]` 动作。
    - buffer 空时再次 replan。

输出：

- 环境实际执行的 action：`[B, action_dim]`。
- 每次 replan 的内部统计：
  - `selected_wm_cost_mean`
  - `selected_model_score_mean`
  - `finite_candidate_rate`
  - `fallback_rate`
  - `avg_generation_time_sec`
  - `avg_scoring_time_sec`
  - `avg_selection_time_sec`
  - `refinement_time_total_sec`
  - `last_refinement_cost_before/after`
  - `last_refinement_goal_cost_before/after`
  - `last_refinement_delta_norm`

### `mpc_cem`: Stable-WorldModel CEM Planner

代码入口：`eval.py` 中 `planner_type == "mpc"` 分支。solver 来自 `stable_worldmodel.solver.CEMSolver`。

输入：

- `AutoCostModel(cfg.policy)`：同一个 LeWM cost model。
- `PlanConfig(horizon=5, receding_horizon=5, action_block=5)`。
- `info_dict`：当前 observation、goal observation、action history 等。
- CEM 默认 solver 参数：
  - `num_samples=300`
  - `n_steps=30`
  - `topk=30`
  - `var_scale=1`

模块和数据流：

1. `WorldModelPolicy` 预处理 `info_dict`，包括 image transform 和 normalizer。
2. solver 初始化 action distribution：
   - `mean: [B, horizon, action_block * action_dim]`
   - `var: [B, horizon, action_block * action_dim]`
3. 每个 CEM iteration：
   - 从高斯分布采样 `candidates: [B, num_samples, horizon, action_block * action_dim]`。
   - 第一个 sample 强制设为当前 mean。
   - 扩展 `info_dict` 到 `[B, num_samples, ...]`。
   - 调用 `model.get_cost(info, candidates)`，通过 world model rollout 得到每个候选的 goal cost。
   - 取 cost 最低的 `topk=30` 个 elite。
   - 更新 `mean = elite.mean(dim=1)`，`var = elite.std(dim=1)`。
4. 30 次迭代结束后输出 `actions = mean`。
5. `WorldModelPolicy` 执行 receding horizon 中的第一个 action block，并在后续需要时重新规划。

输出：

- `actions: [B, horizon, action_block * action_dim]`
- 环境执行动作来自该 action sequence 的前部。
- 统计：
  - `global_planning_calls`
  - `planning_time_total_sec`
  - `avg_planning_time_sec`

### `lgbs_gcidm`: Goal-Conditioned Inverse Dynamics Model

代码入口：`external/latent-geometry-beyond-search/eval_idm.py`，策略类为 `GoalConditionedPolicy`。

输入：

- LeWM/JEPA checkpoint：用于把当前图像和目标图像编码到 latent。
- GC-IDM checkpoint：`GoalConditionedIDM`。
- `info_dict["pixels"]` 和 `info_dict["goal"]`。
- `eval_budget`：用于计算 remaining steps。

模块和数据流：

1. `_prepare_info(info_dict)`：
   - 对像素做 transform。
   - 将 HWC 图像转为 CHW。
   - 对非像素字段按 normalizer 处理。
2. 取当前帧和目标帧：
   - `pixels = info_dict["pixels"][:, -1]`
   - `goal = info_dict["goal"][:, -1]`
3. LeWM/JEPA 编码当前帧：
   - `enc_out = jepa.encoder(pixels, interpolate_pos_encoding=True)`
   - `z_current = jepa.projector(enc_out.last_hidden_state[:, 0])`
4. LeWM/JEPA 编码目标帧：
   - 如果目标图像和上次相同，使用 cached `z_goal`。
   - 否则重新编码目标图像。
5. 计算 remaining horizon：
   - `remaining = min(eval_budget - step_count, idm.max_horizon)`
   - `steps_remaining: [B]`
6. GC-IDM 前向：
   - 输入：`z_current: [B, latent_dim]`，`z_goal: [B, latent_dim]`，`steps_remaining: [B]`
   - 输出：`action: [B, action_dim]`
7. 每个环境 step 都重新执行一次 GC-IDM forward，不做多步候选搜索。

输出：

- 单步 action：`[B, action_dim]`
- 统计：
  - `planning_time_total_sec = sum(idm_policy._plan_times)`
  - `global_planning_calls = len(idm_policy._plan_times)`
  - `effective_replans_per_episode = eval_budget`

这个方法的关键区别是：它没有在线采样候选，也没有 world-model rerank；所有 planning 被 amortize 到一个前向网络里。

### `lgbs_mppi`: MPPI Solver

代码入口：`external/latent-geometry-beyond-search/eval_othersolvers.py --solver mppi`，solver 来自 `stable_worldmodel.solver.MPPISolver`。

输入：

- `AutoCostModel(checkpoint)`。
- `PlanConfig(horizon=5, receding_horizon=5, action_block=5)`。
- `info_dict`。
- MPPI 默认参数：
  - `num_samples=300`
  - `n_steps=30`
  - `topk=30`
  - `var_scale=1.0`
  - `temperature=0.5`

模块和数据流：

1. 初始化 action distribution：
   - `mean: [B, horizon, action_block * action_dim]`
   - `var: [B, horizon, action_block * action_dim]`
2. 每个 MPPI iteration：
   - 采样 noise。
   - 构造 `candidates = mean + noise * var`。
   - 第一个 sample 设为 mean。
   - 调用 world model `get_cost(info, candidates)` 得到 `[B, num_samples]` cost。
   - 取 top-k 低 cost 候选。
   - 用 `softmax(-(cost - min_cost) / temperature)` 得到权重。
   - 更新 `mean = weighted_sum(candidates)`。
   - 标准 MPPI 不更新 `var`。
3. 输出最终 mean action sequence。

输出：

- `actions: [B, horizon, action_block * action_dim]`
- 统计通过 `SolverTimingWrapper` 记录：
  - `planning_time_total_sec`
  - `global_planning_calls`

### `lgbs_icem`: Improved CEM Solver

代码入口：`external/latent-geometry-beyond-search/eval_othersolvers.py --solver icem`，solver 来自 `stable_worldmodel.solver.ICEMSolver`。

输入：

- `AutoCostModel(checkpoint)`。
- `PlanConfig(horizon=5, receding_horizon=5, action_block=5)`。
- `info_dict`。
- iCEM 默认参数：
  - `num_samples=300`
  - `n_steps=30`
  - `topk=30`
  - `var_scale=1`
  - `noise_beta=2.0`
  - `alpha=0.1`
  - `n_elite_keep=5`
  - `return_mean=True`

模块和数据流：

1. 初始化 action distribution：
   - `mean: [B, horizon, action_block * action_dim]`
   - `var: [B, horizon, action_block * action_dim]`
2. 预计算 colored noise 的 FFT scale：
   - `noise_beta=2.0` 让噪声更低频，从而生成更平滑的 action sequence。
3. 每个 iCEM iteration：
   - 采样 white noise。
   - 对时间维做 FFT，按频率缩放，再 inverse FFT，得到 colored noise。
   - 标准化 colored noise。
   - 构造候选：
     `candidates = colored_noise * var + mean`
   - 第一个 sample 设为当前 mean。
   - 将上一轮 elite 中最多 `n_elite_keep=5` 个注入本轮候选。
   - 如果 action bounds 存在，将 candidates clamp 到 action range。
   - 调用 `model.get_cost(info, candidates)`。
   - 取 top-k elite。
   - 用 EMA 更新：
     - `mean = alpha * old_mean + (1 - alpha) * elite_mean`
     - `var = alpha * old_var + (1 - alpha) * elite_var`
4. 如果 `return_mean=True`，输出最终 mean；否则输出最优 elite。

输出：

- `actions: [B, horizon, action_block * action_dim]`
- 统计：
  - `planning_time_total_sec`
  - `avg_planning_time_sec`

相比 CEM，iCEM 的关键改动是 colored noise、elite carryover 和 EMA 更新，因此通常轨迹更平滑，但仍需要大量 world-model rollout。

### `lgbs_gradient`: Gradient-Based Action Optimization

代码入口：`external/latent-geometry-beyond-search/eval_othersolvers.py --solver gradient`，solver 来自 `stable_worldmodel.solver.GradientSolver`。

输入：

- `AutoCostModel(checkpoint)`。
- `PlanConfig(horizon=5, receding_horizon=5, action_block=5)`。
- `info_dict`。
- 当前实验中 `eval_othersolvers.py` 给 GradientSolver 传入：
  - `n_steps=30`
  - `num_samples=2`
  - optimizer 默认 `SGD(lr=1.0)`
  - `var_scale=1`
  - `action_noise=0`

模块和数据流：

1. 初始化可优化动作参数：
   - `actions: [B, num_samples, horizon, action_block * action_dim]`
   - 第一个 sample 从零或上次 init_action 开始。
   - 其他 sample 加随机扰动。
2. 按 batch 处理 env。
3. 每个 gradient step：
   - 将 `info_dict` 扩展到 `[B, num_samples, ...]`。
   - 调用 `model.get_cost(info, batch_init)`。
   - cost 必须保留梯度。
   - 对所有 env/sample 的 cost 求和。
   - 反向传播到 action tensor。
   - 用 SGD 更新 action。
   - 可选加入 `action_noise`，当前实验为 0。
4. 30 步优化结束后：
   - 对每个 env 选择 cost 最低的 sample。
   - 输出该 sample 的 action sequence。

输出：

- `actions: [B, horizon, action_block * action_dim]`
- 统计：
  - `planning_time_total_sec`
  - `avg_planning_time_sec`

这个方法直接对 action 做梯度优化，速度可以比较快，但从当前结果看容易陷入局部解或产生不稳定动作，因此成功率和 jerk 都较差。

### Unified Output Metrics

所有方法最终都会进入同一套评估统计：

- `success_rate`
- `episode_successes`
- `evaluation_time_sec`
- `global_planning_calls`
- `planning_time_total_sec`
- `avg_planning_time_sec`
- `effective_replans_per_episode`
- trajectory quality：
  - `final_goal_distance_mean`
  - `min_goal_distance_mean`
  - `steps_to_success_mean`
  - `path_length_mean`
  - `straight_line_ratio_mean`
  - `action_l2_mean_mean`
  - `action_delta_l2_mean_mean`
  - `action_jerk_l2_mean_mean`

## Overall Summary

| Method | Avg Success | Avg Eval Time | Avg Planning Time | Avg Action Jerk | Overall Position |
| --- | ---: | ---: | ---: | ---: | --- |
| `lgbs_gcidm` | 96.00 | 36.55 | 7.78 | 0.0296 | 跨任务最强的速度和平滑度基线，成功率也最高 |
| `ours_full` | 95.33 | 49.48 | 16.59 | 0.4608 | 成功率接近最优，明显快于搜索式 MPC，但动作平滑性弱于 GC-IDM |
| `mpc_cem` | 82.44 | 832.03 | 800.36 | 1.3572 | 稳定但非常慢，整体成功率低于 ours 和 GC-IDM |
| `lgbs_icem` | 81.89 | 780.81 | 740.84 | 0.4784 | 比 CEM 平滑一些，但速度仍接近搜索式方法，成功率不够稳定 |
| `lgbs_mppi` | 56.39 | 218.92 | 181.44 | 4.2193 | 成功率和轨迹质量都较弱 |
| `lgbs_gradient` | 18.67 | 52.49 | 16.48 | 5.0623 | 虽然快，但成功率很低，不适合作为主要对比方法 |

总体上，`ours_full` 的核心优势是：在四个任务上平均成功率达到 95.33%，显著超过 `mpc_cem` 的 82.44%，同时平均 planning 时间从 800.36s 降到 16.59s。主要短板是动作平滑性，尤其在 `reacher` 上与 `lgbs_gcidm` 差距很大。

`lgbs_gcidm` 是当前最强基线：平均成功率 96.00%，平均 planning 时间 7.78s，平均 action jerk 0.0296。它的优势主要来自直接 amortize policy，不需要像 diffusion rerank 一样生成大量候选并调用 world model 评分。

## Cube

| Method | Success | Eval Time | Planning Time | Final Dist | Steps To Success | Action Jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbs_gcidm` | 99.33 | 67.43 | 9.80 | 0.0232 | 22.68 | 0.0619 |
| `ours_full` | 98.00 | 84.09 | 19.06 | 0.0237 | 10.20 | 0.0880 |
| `lgbs_icem` | 70.00 | 671.75 | 589.80 | 0.0652 | 7.53 | 0.5166 |
| `mpc_cem` | 68.22 | 610.57 | 560.24 | 0.0661 | 4.13 | 1.8799 |
| `lgbs_mppi` | 52.67 | 235.55 | 153.71 | 0.0915 | 1.77 | 5.2417 |
| `lgbs_gradient` | 29.33 | 79.66 | 18.67 | 0.1763 | 0.15 | 3.7854 |

Cube 上 `ours_full` 和 `lgbs_gcidm` 都显著强于搜索式 MPC。`lgbs_gcidm` 成功率最高，且更快、更平滑；`ours_full` 成功率只低 1.33 个百分点，但成功步数更少，说明它在成功 episode 中更早达到成功条件。

相对 `mpc_cem`，`ours_full` 的成功率从 68.22 提升到 98.00，同时 planning speedup 为 29.40x。这个任务上 diffusion planner 的优势非常明确。

## PuShT

| Method | Success | Eval Time | Planning Time | Final Dist | Steps To Success | Action Jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_full` | 91.11 | 33.05 | 17.09 | 64.98 | 20.74 | 0.0474 |
| `mpc_cem` | 87.56 | 930.72 | 913.36 | 120.09 | 22.74 | 0.4221 |
| `lgbs_gcidm` | 84.67 | 23.43 | 7.10 | 80.69 | 34.13 | 0.0246 |
| `lgbs_icem` | 78.89 | 773.77 | 752.01 | 99.84 | 23.61 | 0.1659 |
| `lgbs_mppi` | 57.33 | 208.27 | 188.23 | 285.25 | 26.11 | 1.4314 |
| `lgbs_gradient` | 2.67 | 38.14 | 17.79 | 2006.72 | 10.33 | 2.3971 |

PuShT 是 `ours_full` 最有说服力的任务：成功率 91.11，是所有方法最高；同时 final goal distance 最低，说明不是仅仅碰巧成功，而是整体轨迹更接近目标。

`lgbs_gcidm` 速度和平滑度最好，但成功率只有 84.67，低于 `ours_full` 和 `mpc_cem`。这说明 PuShT 对闭环修正、候选重排和 goal-conditioned world-model scoring 更敏感，纯 amortized action prediction 在这个任务上没有完全压过 diffusion rerank。

相对 `mpc_cem`，`ours_full` 成功率提升 3.56 个百分点，evaluation speedup 为 28.16x，planning speedup 为 53.46x。

## Reacher

| Method | Success | Eval Time | Planning Time | Final Dist | Steps To Success | Action Jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbs_gcidm` | 100.00 | 31.80 | 6.61 | 0.0521 | 41.34 | 0.0013 |
| `ours_full` | 92.22 | 51.60 | 16.75 | 0.0527 | 28.47 | 0.7896 |
| `mpc_cem` | 88.00 | 1074.71 | 1036.75 | 0.0526 | 28.78 | 1.0889 |
| `lgbs_icem` | 83.33 | 864.50 | 828.65 | 0.0596 | 29.12 | 0.4759 |
| `lgbs_mppi` | 42.89 | 214.40 | 184.84 | 0.2252 | 25.59 | 3.5436 |
| `lgbs_gradient` | 6.67 | 48.16 | 15.52 | 4.7063 | 1.92 | 4.0276 |

Reacher 上 `lgbs_gcidm` 明显领先：100% 成功率、最低耗时、最低 jerk。`ours_full` 是第二名，成功率 92.22，高于 `mpc_cem` 的 88.00，并且 planning speedup 达到 61.89x。

但 Reacher 也是 `ours_full` 当前最明显的短板。它的 final distance 与 `lgbs_gcidm`、`mpc_cem` 接近，说明 goal ranking 并非完全失效；真正的问题更像是 action chunk 的时序平滑性和可执行性。`ours_full` 的 action jerk 是 0.7896，而 `lgbs_gcidm` 只有 0.0013，差距非常大。这与之前观察到的 rollout 跳变现象一致。

后续如果只优化 PuShT 和 Reacher，Reacher 应优先考虑 action temporal consistency、smoothness regularization、chunk-level refinement 或用 learned policy proposal 减少 diffusion chunk 内部突变。

## TwoRoom

| Method | Success | Eval Time | Planning Time | Final Dist | Steps To Success | Action Jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbs_gcidm` | 100.00 | 23.54 | 7.61 | 14.97 | 24.79 | 0.0308 |
| `ours_full` | 100.00 | 29.18 | 13.48 | 13.96 | 10.96 | 0.9181 |
| `lgbs_icem` | 95.33 | 813.21 | 792.91 | 18.17 | 13.42 | 0.7551 |
| `mpc_cem` | 86.00 | 712.13 | 691.09 | 20.47 | 16.89 | 2.0378 |
| `lgbs_mppi` | 72.67 | 217.47 | 198.97 | 20.77 | 19.65 | 6.6606 |
| `lgbs_gradient` | 36.00 | 44.00 | 13.92 | 72.43 | 4.71 | 10.0393 |

TwoRoom 上 `ours_full` 和 `lgbs_gcidm` 都达到 100% 成功率。`lgbs_gcidm` 更快、更平滑；`ours_full` 的 final distance 更低，steps to success 更少，说明 diffusion planner 的候选重排更偏向快速到达目标。

相对 `mpc_cem`，`ours_full` 成功率从 86.00 提升到 100.00，evaluation speedup 为 24.40x，planning speedup 为 51.25x。这个任务上可以把 `ours_full` 描述为成功率和目标推进最强，但平滑性弱于 `lgbs_gcidm`。

## Method-Level Takeaways

`ours_full` 是当前最值得主推的方法。它在 PuShT 上取得最高成功率，在 TwoRoom 上达到 100%，在 Cube 上接近最优，在 Reacher 上排名第二。相对 `mpc_cem`，它在所有任务上都显著更快，并且通常更成功。

`lgbs_gcidm` 是最强 baseline。它在 Cube、Reacher、TwoRoom 上最强或并列最强，并且几乎所有任务都具有最低 planning time 和最低 action jerk。它的主要弱点在 PuShT，成功率低于 `ours_full`。

`mpc_cem` 适合作为传统搜索 baseline，但不适合作为效率 baseline。它耗时远高于 amortized 或 diffusion planner，并且在 Cube、Reacher、TwoRoom 上成功率都低于 `ours_full`。

`lgbs_icem` 比 `mpc_cem` 在部分任务上成功率更高或平滑性更好，但耗时仍然很大，整体不是主要竞争者。

`lgbs_mppi` 和 `lgbs_gradient` 当前结果不稳定。尤其 `lgbs_gradient` 虽然时间短，但成功率过低、jerk 很高，不应作为强 baseline，只适合作为补充对比。

## Main Conclusions

1. `ours_full` 已经显著优于传统 `mpc_cem`：平均成功率 95.33 vs 82.44，平均 planning time 16.59s vs 800.36s。
2. 与 `lgbs_gcidm` 相比，`ours_full` 的成功率非常接近：95.33 vs 96.00，但速度和平滑性仍落后。
3. PuShT 是 `ours_full` 的优势任务：成功率最高，并且 final goal distance 最低。
4. Reacher 是当前最需要优化的任务：`ours_full` 成功率 92.22，低于 `lgbs_gcidm` 的 100.00；主要差距不在 final distance，而在 action jerk 和 chunk 可执行性。
5. TwoRoom 和 Cube 说明 diffusion action head + world-model rerank 是有效的：高成功率、低规划耗时，并且明显优于搜索式 MPC。

## Next Steps

优先优化 Reacher 和 PuShT 时，建议保留 diffusion action head，将重点放在动作序列质量上：

- 在 diffusion 训练中加入 action smoothness / jerk regularization，减少 chunk 内部跳变。
- 在 inference 阶段加入轻量 temporal refinement，只修正候选动作的平滑性，不破坏 world-model goal ranking。
- 尝试 learned proposal 或 GC-IDM proposal + diffusion refinement，用 amortized policy 提供更平滑初值，再由 diffusion planner 保留多模态候选能力。
- 对 Reacher 单独分析失败 episode：比较成功/失败轨迹的 jerk、action delta、selected wm cost，确认失败是否集中在高频动作突变。
