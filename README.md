# LeWM Diffusion Planner

本仓库当前主线是把 LeWM world model 上的在线规划，整理成可复现的
diffusion action-chunk planner。代码保留三条需要继续系统测试的路径：

1. 已经大规模测试过的 diffusion 主线模型。
2. Reacher H=10 的 `top16-wmcost` score-head 筛选模型。
3. Reacher 上不同 action horizon `H` 的 backbone 训练与评估。

核心流程如下：

```text
raw HDF5 demonstrations
  -> planner dataset: z_cur, z_goal, teacher_plan
  -> K-means action anchors
  -> anchor-conditioned truncated diffusion backbone
  -> optional score-head fine-tuning
  -> eval.py closed-loop evaluation
```

## 目录结构

```text
config/eval/                         # eval.py 的任务配置和 profile
diffusion/                           # diffusion 模型、训练 loss、runtime policy
evaluation/                          # trajectory quality 等评估辅助逻辑
planners/                            # planner dataset 和 action anchor 构建入口
scripts/run_reacher_diffusion_horizon_pipeline.sh
                                      # Reacher H=5/10/15/... backbone pipeline
train_diffusion_planner.py           # diffusion backbone / score-head 训练入口
eval.py                              # 闭环评估入口
```

`config/eval/<task>.yaml` 现在同时管理同一个任务的环境配置、dataset 路径和
planner profiles。运行时用 `eval_profile=...` 选择 profile，避免同一组参数散落在多个
alias 配置里。

## 1. Diffusion 主线模型

主线 diffusion planner 使用 `K=128` 个 action anchors，先生成 128 个候选
action chunk，再用 LeWM world-model cost 在候选里选最优动作。这个路径对应 runtime
selection mode：

```text
diffusion_selection_mode=wm_only
diffusion_num_candidates=128
```

Reacher 当前主线 bundle 写在 `config/eval/reacher.yaml` 的 `profiles.diffusion`：

```text
/data/ykz/reacher/diffusion_pipeline/reacher_diffusion_k128_200000/diffusion_planner_best_bundle.pt
```

评估命令：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion
```

小规模 smoke test 可以缩短 episode 数：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion \
  eval.num_eval=1 \
  eval.eval_budget=25 \
  trajectory_quality.enabled=false
```

评估时间统一看这些字段：

```text
[summary] evaluation_time=...
[planner-stats] global_planning_calls=...
[planner-stats] planning_time_total_sec=...
[planner-stats] avg_planning_time_sec=...
[worldmodel-stats] wm_rollout_candidate_count=...
```

其中 `evaluation_time` 是整个 `world.evaluate_from_dataset(...)` 的 wall-clock 时间；
不同 planner 的规划速度比较优先看 `planning_time_total_sec` 和
`avg_planning_time_sec`。如果涉及 `wm_only` 或 `score_topk_wm`，还要同时看
`wm_rollout_candidate_count`，确认实际计算了多少个 world-model candidate。

## 2. H=10 Top16-WMCost Score Head

`top16-wmcost` 的目标不是完全替代 world-model cost，而是把 128 个 diffusion 候选先
用 score head 预筛成 16 个，再只在这 16 个里面计算 wm-cost：

```text
generate 128 candidates
  -> score_head rank
  -> keep score top16
  -> compute wm-cost for top16 only
  -> select the minimum wm-cost candidate inside top16
```

对应 runtime profile 是 `diffusion_h10_score_top16_wm`：

```text
diffusion_bundle=/data/ykz/reacher/diffusion_pipeline/reacher_h10_score_head_mlp_top16_margin/diffusion_planner_best_bundle.pt
diffusion_selection_mode=score_topk_wm
diffusion_score_topk=16
diffusion_num_candidates=128
diffusion_runtime_execute_steps=10
plan_config.receding_horizon=2
plan_config.action_block=5
```

评估命令：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion_h10_score_top16_wm
```

smoke test：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion_h10_score_top16_wm \
  eval.num_eval=1 \
  eval.eval_budget=25 \
  trajectory_quality.enabled=false
```

日志里应该能看到：

```text
selection_mode=score_topk_wm score_topk=16
[diffusion-rerank] finite_candidate_rate=0.1250
[worldmodel-stats] wm_rollout_candidate_count=<global_planning_calls * 16>
```

这三项一起确认 runtime 的确只对 top16 计算 wm-cost。

### Score Head 训练

当前 score head 使用 MLP：

```text
LayerNorm(hidden_dim)
  -> Linear(hidden_dim, score_head_hidden_dim)
  -> activation
  -> ... repeated score_head_num_layers times
  -> Linear(score_head_hidden_dim, 1)
```

训练时冻结 diffusion backbone，只训练 `score_head.*`。loss 用
`wm_score_topk_margin` preset：

```text
rec / BCE / anchor score loss: disabled
wm-score target: minimum wm-cost candidate as top-1 CE label
top-k margin: push the minimum wm-cost candidate above the top16 boundary
candidate source: inference candidates, matching eval-time denoising candidates
```

Reacher H=10 score-head 训练命令：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u train_diffusion_planner.py \
  --dataset-path /data/ykz/reacher/diffusion_pipeline/single_peak_reacher_traj_h10_200k_raw.pt \
  --anchor-bundle-path /data/ykz/reacher/diffusion_pipeline/reacher_action_anchors_h10_200k_k128_raw.pt \
  --init-bundle-path /data/ykz/reacher/diffusion_pipeline/reacher_h10_diffusion_200k_simple_bce_k128_raw/diffusion_planner_best_bundle.pt \
  --wm-policy /data/ykz/reacher/lewm_epoch_29 \
  --output-dir /data/ykz/reacher/diffusion_pipeline/reacher_h10_score_head_mlp_top16_margin \
  --device cuda \
  --epochs 80 \
  --batch-size 64 \
  --val-batch-size 128 \
  --num-workers 4 \
  --loss-preset wm_score_topk_margin \
  --score-head-type mlp \
  --score-head-hidden-dim 256 \
  --score-head-num-layers 2 \
  --freeze-non-score-head \
  --wm-score-topk-margin-k 16 \
  --wm-score-topk-margin 0.1 \
  --wm-score-topk-margin-weight 0.5 \
  --log-every 50
```

训练时主要看：

```text
train_wm_score_acc / val_wm_score_acc
train_wm_score_topk_acc / val_wm_score_topk_acc
train_wm_score_ranking_loss / val_wm_score_ranking_loss
```

`top16-wmcost` 需要的是 `wm_score_topk_acc` 稳定较高；`wm_score_acc` 直接对应 top1，
通常更难提升。

## 3. Action Horizon H 探索

H 表示 diffusion action chunk 覆盖的环境 step 数。Reacher 当前约定固定：

```text
action_block=5
receding_horizon=H / action_block
```

所以 `H` 必须是 5 的倍数。我们用同一个脚本控制 H，其他参数保持主线版本：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
scripts/run_reacher_diffusion_horizon_pipeline.sh \
  --horizon 5 \
  --output-root /data/ykz/reacher/diffusion_pipeline \
  --device cuda

CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
scripts/run_reacher_diffusion_horizon_pipeline.sh \
  --horizon 10 \
  --output-root /data/ykz/reacher/diffusion_pipeline \
  --device cuda

CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
scripts/run_reacher_diffusion_horizon_pipeline.sh \
  --horizon 15 \
  --output-root /data/ykz/reacher/diffusion_pipeline \
  --device cuda
```

这个 pipeline 只训练 diffusion backbone，不微调 score head。它会依次执行：

```text
planners/build_single_peak_dataset.py
planners/build_action_anchors.py
train_diffusion_planner.py --loss-preset simple_bce
```

默认输出路径模板：

```text
/data/ykz/reacher/diffusion_pipeline/single_peak_reacher_traj_h<H>_200k_raw.pt
/data/ykz/reacher/diffusion_pipeline/reacher_action_anchors_h<H>_200k_k128_raw.pt
/data/ykz/reacher/diffusion_pipeline/reacher_h<H>_diffusion_200k_simple_bce_k128_raw/diffusion_planner_best_bundle.pt
/data/ykz/reacher/diffusion_pipeline/reacher_h<H>_diffusion_200k_simple_bce_k128_raw/pipeline_logs/
/data/ykz/reacher/diffusion_pipeline/reacher_h<H>_diffusion_200k_simple_bce_k128_raw/pipeline_summary.txt
```

H=10 已有 profile：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion_h10_wm_only
```

如果临时评估 H=5 或 H=15，而还没有写入固定 profile，可以在 `diffusion` profile 上覆盖
bundle 和 runtime H：

```bash
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python -u eval.py \
  --config-name reacher \
  eval_profile=diffusion \
  profiles.diffusion.diffusion_bundle=/data/ykz/reacher/diffusion_pipeline/reacher_h15_diffusion_200k_simple_bce_k128_raw/diffusion_planner_best_bundle.pt \
  plan_config.receding_horizon=3 \
  plan_config.action_block=5 \
  diffusion_runtime_execute_steps=15
```

H=5 时把 bundle 改成 `reacher_h5_...`，并设置：

```text
plan_config.receding_horizon=1
diffusion_runtime_execute_steps=5
```

## 4. 系统测试建议

后续系统测试优先固定同一张 GPU、同一个 `eval.num_eval` 和同一个
`eval.eval_budget`，比较下面三组：

```text
1. eval_profile=diffusion
   25-action diffusion + 128-candidate wm_only

2. eval_profile=diffusion_h10_wm_only
   10-action diffusion + 128-candidate wm_only

3. eval_profile=diffusion_h10_score_top16_wm
   10-action diffusion + score top16 prefilter + wm-cost
```

每次记录：

```text
success_rate
evaluation_time
planning_time_total_sec
avg_planning_time_sec
global_planning_calls
wm_rollout_candidate_count
wm_rollout_time_total_sec
```

速度比较以 `planning_time_total_sec` / `avg_planning_time_sec` 为主；
`evaluation_time` 会包含环境 reset、渲染、dataset replay 等闭环开销，只作为整体 wall-clock
参考。

## 5. 基本验证

配置和 top16 runtime 相关单测：

```bash
./.venv/bin/python -m pytest \
  tests/test_eval_planner_configs.py::EvalPlannerConfigTests::test_reacher_h10_eval_profiles_resolve_runtime_overrides \
  tests/test_diffusion_policy_refinement.py::DiffusionPolicyRefinementTest::test_score_topk_wm_scores_only_score_prefiltered_candidates \
  -q
```

全量测试：

```bash
./.venv/bin/python -m pytest -q
```
