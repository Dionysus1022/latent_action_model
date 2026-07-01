# Comparison Experiment Plan

本文档用于统一第一阶段比较实验。当前阶段只做 method-level comparison，不做参数消融。

## 核心策略

第一阶段只比较三类完整方法：

1. **Ours: Diffusion Planner + 全部优化**
   当前 diffusion planner 作为我们的完整系统评估。评估时打开已经确认有用的优化，例如 world-model rerank 和 refinement。它不是消融对象，而是一个完整方法。

2. **Baseline: MPC / CEM**
   使用 LeWM world model + CEM solver 的在线搜索方法，作为传统 planning baseline。

3. **Paper Reproduction: GC-IDM**
   复现 `Latent Geometry Beyond Search: Amortizing Planning in World Models` 的 GC-IDM 方法，作为 amortized planning baseline。

不在第一阶段 sweep threshold、candidate 数、anchor 方式、runtime execute steps、denoise steps 等参数。那些属于第二阶段诊断或消融实验。

## 实验目标

第一阶段要回答：

1. 我们的 full diffusion planner 相比 MPC / CEM 是否能保持或提升成功率。
2. 我们的 full diffusion planner 相比 MPC / CEM 是否更快。
3. 我们的 full diffusion planner 相比 GC-IDM 是否更强，尤其是在成功率、速度、轨迹质量上。
4. 如果 Reacher / PushT 仍然失败，先只记录现象，不在主实验里临时改参数。

## 固定评估协议

所有主比较实验固定：

- `eval.num_eval=50`
- `eval.goal_offset_steps=25`
- `eval.eval_budget=50`
- seeds: `42, 43, 44`
- repeats per seed: `0, 1, 2`
- 同一个 task 固定同一个 dataset
- 同一个 task 固定同一个 world model checkpoint
- 同一个 method 固定同一个 planner bundle
- 所有主实验打开 `trajectory_quality.enabled=true`
- 批量实验默认 `trajectory_quality.save_video=false`
- debug 时才打开 video

实验规模：

```text
4 tasks x 6 methods x 3 seeds x 3 repeats = 216 eval runs
216 runs x 50 episodes = 10800 evaluated episodes
```

统计口径：

- 同一个 `task x method x seed` 的 3 次 repeat 先求均值和标准差。
- 再对 3 个 seed 聚合，得到最终 `task x method` 的 mean / std。
- 不要把 9 次 run 直接当成 9 个独立 seed；同一 seed 对应同一批 episode starts，repeat 主要衡量 planner 随机性、GPU 非确定性和耗时波动。

正式结果按 `task x method` 汇总 mean / std，并写入 `docs/experiments/result.md`。

## 实验矩阵

第一阶段主仓库 comparison pipeline 总共 216 次 eval：

| Dimension | Values |
| --- | --- |
| Tasks | `cube`, `pusht`, `reacher`, `tworoom` |
| Methods | `mpc_cem`, `ours_full`, `lgbs_gcidm`, `lgbs_mppi`, `lgbs_icem`, `lgbs_gradient` |
| Seeds | `42`, `43`, `44` |
| Repeats | `0`, `1`, `2` |

`repeat` 不改变 `seed`。它只记录同一配置重复运行的编号。命令中仍然使用对应的 `seed=<seed>`。

## 数据集和任务

第一阶段覆盖四个任务：

| Task | Config | Dataset |
| --- | --- | --- |
| `cube` | `--config-name cube` | `/data/ykz/cube/cube_single_expert.h5` |
| `pusht` | `--config-name pusht` | `/data/ykz/pusht/pusht_expert_train.h5` |
| `reacher` | `--config-name reacher` | `/data/ykz/reacher/reacher.h5` |
| `tworoom` | `--config-name tworoom` | `/data/ykz/tworoom/tworoom.h5` |

如果实际文件路径不同，以本机路径为准，但必须写入结果记录。

## 第一阶段方法表

| Method Name | Eval profile / Overrides | 说明 |
| --- | --- | --- |
| `mpc_cem` | `eval_profile=mpc` | LeWM + CEM online search baseline |
| `ours_full` | task-specific full diffusion config | 我们的完整方法，打开当前全部优化 |
| `lgbs_gcidm` | external `eval_idm.py` | LGBS 论文的 GC-IDM amortized planner |
| `lgbs_mppi` | external `eval_othersolvers.py --solver mppi` | LGBS 论文的 MPPI baseline |
| `lgbs_icem` | external `eval_othersolvers.py --solver icem` | LGBS 论文的 iCEM baseline |
| `lgbs_gradient` | external `eval_othersolvers.py --solver gradient` | LGBS 论文的 gradient planner baseline |

LGBS 系列不通过本仓库 `eval.py` profile 运行；comparison pipeline 会直接调用
`external/latent-geometry-beyond-search/eval_idm.py` 或 `eval_othersolvers.py`，
并把统一 `[summary]`、`[planner-stats]`、`[trajectory-quality]` 输出并入同一张结果表。

`ours_full` 不等价于裸 `eval_profile=diffusion`。它应该使用每个任务当前最强、最合理的 full-system 配置。

建议定义如下：

| Task | `ours_full` 推荐配置 |
| --- | --- |
| `cube` | `eval_profile=diffusion diffusion_selection_mode=wm_only diffusion_refinement.enabled=true` |
| `tworoom` | `eval_profile=diffusion diffusion_selection_mode=wm_only diffusion_refinement.enabled=true` |
| `reacher` | `eval_profile=diffusion diffusion_selection_mode=wm_only diffusion_refinement.enabled=true` |
| `pusht` | `eval_profile=diffusion diffusion_selection_mode=wm_only diffusion_refinement.enabled=true` |

注意：一旦选定 `ours_full` 配置，第一阶段内不要再改。若要换配置，需要重新跑所有 seed，并把旧结果标记为废弃或 exploratory。

## 命令模板

### 一键完整实验

运行完整第一阶段比较实验：

```bash
scripts/run_comparison_experiments.sh
```

默认会执行：

```text
4 tasks x 6 methods x 3 seeds x 3 repeats = 216 eval runs
```

脚本会显示总进度条，当前进度会包含 `task/method/seed/repeat`。输出位置：

```text
outputs/comparison_experiments/
outputs/comparison_experiments/raw_runs.csv
outputs/comparison_experiments/seed_summary.csv
outputs/comparison_experiments/final_summary.csv
docs/experiments/result.md
```

先检查命令矩阵，不真正运行 eval：

```bash
scripts/run_comparison_experiments.sh --dry-run --force
```

只跑某个任务或方法做 smoke test：

```bash
scripts/run_comparison_experiments.sh \
  --tasks reacher \
  --methods ours_full \
  --seeds 42 \
  --repeats 0
```

### MPC / CEM

```bash
./.venv/bin/python eval.py \
  --config-name <task> \
  eval_profile=mpc \
  seed=<seed> \
  eval.num_eval=50 \
  trajectory_quality.enabled=true \
  trajectory_quality.save_video=false
```

### LGBS Paper Reproduction / Baselines

```bash
scripts/run_lgbs_pipeline.sh --task <task> --stage all --seed <seed>
scripts/run_comparison_experiments.sh --methods lgbs_gcidm,lgbs_mppi,lgbs_icem,lgbs_gradient
```

注意：`lgbs_gcidm` 评估前必须已经有
`/data/ykz/lgbs_repro/<task>/<task>_gcidm.pt`。如果不存在，先运行
`scripts/run_lgbs_pipeline.sh --task <task> --stage train` 或 `--stage all`。

### Ours Full: Cube / TwoRoom / Reacher

```bash
./.venv/bin/python eval.py \
  --config-name <task> \
  eval_profile=diffusion \
  diffusion_selection_mode=wm_only \
  diffusion_refinement.enabled=true \
  seed=<seed> \
  eval.num_eval=50 \
  trajectory_quality.enabled=true \
  trajectory_quality.save_video=false
```

### Ours Full: PushT

```bash
./.venv/bin/python eval.py \
  --config-name pusht \
  eval_profile=diffusion \
  diffusion_selection_mode=wm_only \
  diffusion_refinement.enabled=true \
  seed=<seed> \
  eval.num_eval=50 \
  trajectory_quality.enabled=true \
  trajectory_quality.save_video=false
```

## 必须记录的主指标

每个 `task x method x seed` 记录：

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 成功率 |
| `episode_successes` | 每个 episode 是否成功 |
| `evaluation_time_sec` | 整次评估耗时 |
| `planning_time_total_sec` | planner 总耗时 |
| `avg_planning_time_sec` | 单次 planning 平均耗时 |
| `global_planning_calls` | planning 调用次数 |
| `effective_replans_per_episode` | 每个 episode 的有效 replan 次数 |

汇总时额外计算：

- `success_rate_mean`
- `success_rate_std`
- `evaluation_time_sec_mean`
- `evaluation_time_sec_std`
- `speedup_vs_mpc`
- `planning_speedup_vs_mpc`

## Trajectory Quality 指标

每次主实验都记录：

| 指标 | 含义 |
| --- | --- |
| `final_goal_distance_mean` | 结束时到目标距离 |
| `min_goal_distance_mean` | episode 内最近目标距离 |
| `steps_to_success_mean` | 首次成功步数 |
| `path_length_mean` | 状态轨迹长度 |
| `straight_line_ratio_mean` | 路径绕远程度 |
| `action_l2_mean_mean` | 动作幅值 |
| `action_delta_l2_mean_mean` | 动作一阶变化 |
| `action_jerk_l2_mean_mean` | 动作二阶变化，衡量平滑性 |
| `latent_monotonicity_mean` | latent 到目标距离全程逐步不增加的 episode 比例，和 LGBS Table 4 的 latent monotonicity 对齐 |
| `latent_monotonic_step_fraction_mean` | 每个 episode 内单调靠近 goal 的 step 比例，仅作诊断 |
| `final_latent_goal_distance_mean` | 结束时 latent 到 goal latent 的距离 |
| `min_latent_goal_distance_mean` | episode 内最近 goal latent 距离 |

这部分用于和论文里的 trajectory quality / smoothness 口径对齐。LGBS Table 4
明确报告 `Action Jerk` 和 `Latent Monotonicity`；我们的 `action_jerk_l2_mean_mean`
对应 action jerk，`latent_monotonicity_mean` 对应 latent monotonicity。
成功率相同的时候，优先比较速度、action jerk 和 latent monotonicity。

## Diffusion 额外指标

Ours full 需要额外记录：

- `diffusion_bundle`
- `diffusion_selection_mode`
- `diffusion_num_candidates`
- `diffusion_truncation_steps`
- `diffusion_start_timestep`
- `diffusion_runtime_execute_steps`
- `diffusion_refinement.enabled`
- `avg_generation_time_sec`
- `avg_scoring_time_sec`
- `avg_selection_time_sec`
- `finite_candidate_rate`
- `fallback_rate`
- `selected_wm_cost_mean`
- `selected_model_score_mean`

这些指标不用于第一阶段调参，只用于解释 full method 的行为。

## 训练产物记录

每个 planner bundle 必须记录：

- task
- method name
- world model checkpoint path
- planner bundle path
- dataset path
- training sample count
- training seed
- checkpoint epoch
- checkpoint selection rule: `best` / `last`
- anchor construction method
- diffusion planner 训练配置
- GC-IDM 论文复现命令和训练配置
- 完整 eval command
- log path

没有这些信息的结果只能算 exploratory，不能进入主表。

## 结果表格式

每行是一组 `task x method x seed x repeat`：

```text
task
method
seed
repeat
dataset_h5
wm_policy
planner_bundle
success_rate
evaluation_time_sec
planning_time_total_sec
avg_planning_time_sec
global_planning_calls
effective_replans_per_episode
final_goal_distance_mean
min_goal_distance_mean
steps_to_success_mean
path_length_mean
straight_line_ratio_mean
action_l2_mean_mean
action_delta_l2_mean_mean
action_jerk_l2_mean_mean
diffusion_refinement_enabled
diffusion_runtime_execute_steps
diffusion_num_candidates
diffusion_truncation_steps
command
log_path
```

`docs/experiments/result.md` 同时保留 raw run 表和 summary 表。最终主表按 `task x method` 聚合 mean / std。

## 第一阶段结论口径

报告顺序：

1. 四个任务的 success rate mean / std。
2. 相对 MPC / CEM 的 speedup。
3. 相对 GC-IDM 的 success rate、speed、trajectory quality。
4. action jerk 和 final / min goal distance。
5. 失败 case 只做定性描述，不在第一阶段引入消融结论。

第一阶段可以得出的结论是：

- `ours_full` 是否强于 MPC / CEM。
- `ours_full` 是否强于 GC-IDM 复现。
- 提升来自成功率、速度、轨迹质量中的哪一类。

第一阶段不能得出的结论是：

- 哪个 threshold 最好。
- 哪个 candidate 数最好。
- 哪个 anchor 方式最好。
- runtime execute steps 应该是多少。

这些放到第二阶段消融。

## 第二阶段：后置消融

只有当第一阶段主比较完成后，才进入消融。消融只针对第一阶段暴露出的失败任务。

候选消融包括：

- `diffusion_runtime_execute_steps`
- `diffusion_num_candidates`
- `diffusion_truncation_steps`
- anchor construction
- `diffusion_refinement.enabled`

消融结果单独成表，不和第一阶段主比较表混合。

## 和论文的对齐

如果要和 `Latent Geometry Beyond Search: Amortizing Planning in World Models` 对齐，第一阶段至少报告：

- success rate
- planning time / speedup
- action jerk
- goal-directed trajectory quality
- GC-IDM reproduced result
- MPC / CEM baseline result

只比较成功率是不够的。我们的主张应该建立在成功率、速度和轨迹质量三个维度上。
