# LeWM Diffusion Planner

本仓库基于 LeWorldModel (LeWM)，当前重点是把原本较慢的 CEM/MPC test-time planning
蒸馏成可学习的 action-chunk planner。主要流程是：

```text
raw HDF5 demonstrations
  -> LeWM teacher rollout / CEM planner labels
  -> planner dataset: z_cur, z_goal, teacher_plan
  -> action anchor K-means
  -> anchor-conditioned truncated diffusion planner
  -> eval.py closed-loop evaluation
  -> optional consistency distillation for 1-2 step inference
```

当前支持任务：

- PushT
- TwoRoom
- Reacher
- Cube

## 环境

项目默认使用本地虚拟环境：

```bash
./.venv/bin/python -m pytest --version
```

如果需要重新安装依赖：

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

常用依赖包括 Hydra/OmegaConf、PyTorch、stable_worldmodel、scikit-learn 等。

## 目录结构

```text
config/diffusion/              # diffusion planner 训练 pipeline 的 Hydra 配置
config/eval/                   # eval.py 的 Hydra 配置
config/consistency/            # consistency distillation 的 Hydra 配置
diffusion/                     # diffusion planner 模型、训练、policy、pipeline
planners/                      # teacher dataset / anchors / legacy planner utilities
scripts/train_diffusion_head.py # 一键构建 dataset + anchors + diffusion planner
train_diffusion_planner.py      # 只训练 diffusion planner 最后一步
train_consistency_planner.py    # consistency distillation 入口
eval.py                        # 环境评估入口
```

## 数据处理 Pipeline

### 输入

每个任务需要两个核心输入：

1. 原始 HDF5 demonstration 数据。
2. 已训练好的 LeWM world model checkpoint，即 `task.wm_policy`。

默认路径写在：

```text
config/diffusion/task/cube.yaml
config/diffusion/task/pusht.yaml
config/diffusion/task/reacher.yaml
config/diffusion/task/tworoom.yaml
```

例如 TwoRoom：

```yaml
raw_h5: /data/ykz/tworoom/tworoom.h5
wm_policy: /data/ykz/tworoom/lewm_epoch_67
planner_dataset_path: ${pipeline.output_root}/tworoom_planner_dataset.pt
anchor_bundle_path: ${pipeline.output_root}/tworoom_action_anchors_k${anchors.num_anchors}.pt
train_output_dir: ${pipeline.output_root}/tworoom_diffusion_k${anchors.num_anchors}_${pipeline.num_samples}
```

### 输出

pipeline 会生成：

```text
<task>_planner_dataset.pt
<task>_action_anchors_k<K>.pt
<task>_diffusion_k<K>_<N>/diffusion_planner_best_bundle.pt
<task>_diffusion_k<K>_<N>/diffusion_planner_last_bundle.pt
<task>_diffusion_k<K>_<N>/diffusion_planner_train_summary.pt
```

planner dataset 的核心字段是：

```text
z_cur         # 当前 latent state
z_goal        # goal/subgoal latent state
teacher_plan  # teacher 产生的 action chunk
meta
build_info
```

action anchor bundle 是对 `teacher_plan` 做 K-means 后得到的典型动作块集合。

## 一键训练 Diffusion Planner

推荐入口：

```bash
./.venv/bin/python scripts/train_diffusion_head.py task=tworoom
```

这会按顺序执行：

1. `diffusion.dataset_builder`：从原始 HDF5 + LeWM checkpoint 构建 planner dataset。
2. `diffusion.anchor_builder`：从 `teacher_plan` 中聚类动作锚点。
3. `diffusion.train`：训练 anchor-conditioned diffusion planner。

先 dry-run 检查路径和命令：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  pipeline.device=cpu \
  pipeline.output_root=/tmp/tworoom_diffusion_dry_run \
  pipeline.dry_run=true
```

注意：当前 dry-run 仍会写 `pipeline_summary.yaml`，因此 `pipeline.output_root`
必须指向可写目录。正式训练时再使用 `/data/ykz/...` 这类长期保存路径。

小规模 smoke run：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  pipeline.device=cuda \
  pipeline.num_samples=1000 \
  anchors.num_anchors=16 \
  train.epochs=2 \
  train.batch_size=16 \
  train.val_batch_size=32
```

正式训练示例：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  pipeline.device=cuda \
  pipeline.num_samples=200000 \
  anchors.num_anchors=128 \
  train.epochs=80 \
  train.batch_size=64 \
  train.val_batch_size=128
```

覆盖 LeWM checkpoint：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  task.wm_policy=/data/ykz/tworoom/lewm_epoch_67
```

默认 `pipeline.use_raw_dataset=true`，直接从原始 HDF5 构建 planner dataset。如果要恢复旧的 split-first 方式：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  pipeline.use_raw_dataset=false
```

## 评估 Diffusion Planner

`eval.py` 是 Hydra 入口。任务基础配置在：

```text
config/eval/cube.yaml
config/eval/pusht.yaml
config/eval/reacher.yaml
config/eval/tworoom.yaml
```

这些任务配置现在收敛为一个文件管理本任务的 eval variants：环境、dataset、
reset callables、`plan_config` 仍在顶层；planner/checkpoint/bundle 参数放在同文件的
`profiles` 下面。运行时用 `eval_profile` 选择具体实验：

```text
eval_profile=mpc                 # LeWM + CEM/MPC baseline
eval_profile=diffusion           # multi-step diffusion planner
eval_profile=consistency         # distilled 1-step consistency planner
eval_profile=gc_idm              # GC-IDM closed-loop one-step inverse dynamics
eval_profile=corrective_replan   # PushT Phase2 error-triggered replan
eval_profile=corrective_learned  # PushT Phase3 learned corrector
```

旧的 `config/eval/<task>_mpc.yaml`、`<task>_diffusion.yaml`、
`<task>_consistency.yaml` 和 `pusht_diffusion_corrective.yaml` 已归档到
`config/eval/legacy/`，主目录不再把这些薄 alias 混在任务配置旁边。新实验只改
`<task>.yaml` 里的 profile，避免同一组参数散落在多个文件。

为了兼容旧脚本，`eval.py` 仍会把旧 config name 自动转换为新 profile：

```text
--config-name cube_mpc                    -> --config-name cube eval_profile=mpc
--config-name cube_diffusion              -> --config-name cube eval_profile=diffusion
--config-name cube_consistency            -> --config-name cube eval_profile=consistency
--config-name pusht_diffusion_corrective  -> --config-name pusht eval_profile=corrective_learned
```

运行 MPC baseline：

```bash
./.venv/bin/python eval.py --config-name cube eval_profile=mpc
```

运行 diffusion planner：

```bash
./.venv/bin/python eval.py --config-name cube eval_profile=diffusion
```

运行 GC-IDM：

```bash
./.venv/bin/python eval.py --config-name reacher eval_profile=gc_idm \
  +dataset_h5=/data/ykz/reacher/reacher.h5
```

一键复现 GC-IDM pipeline（构建 `(z_t, z_{t+h}, h) -> a_t` 数据、训练、评估）：

```bash
scripts/run_gc_idm_pipeline.sh \
  --task reacher \
  --input-h5 /data/ykz/reacher/reacher.h5 \
  --wm-policy /data/ykz/reacher/lewm_epoch_29 \
  --num-samples 200000 \
  --epochs 50
```

大数据构建和训练阶段会显示 tqdm 进度条；训练默认按论文配置使用
`lr=1e-3`、cosine annealing 到 `lr/100`、batch size 1024、50 epochs，
episode-level validation split，并默认让 `gc_idm_best_bundle.pt` 保存最后一个
epoch 的 checkpoint。若要使用 validation-best checkpoint，可以在 pipeline 中加
`--checkpoint-selection best`。训练结束会输出 horizon 分桶的 validation MSE/R2。
旧 bundle 如果是用
`lr=1e-4`、batch size 256 训练的，通常会出现 action 预测接近 zero-action
baseline、Reacher 成功率偏低的现象，需要重新训练。

每一步日志保存在
`/data/ykz/reacher/gc_idm/pipeline_logs/`。正式 bundle 默认输出到
`/data/ykz/reacher/gc_idm/gc_idm_best_bundle.pt`，也就是
`config/eval/reacher.yaml` 里 `profiles.gc_idm.gc_idm_bundle` 的默认路径。
先检查命令而不运行：

```bash
scripts/run_gc_idm_pipeline.sh \
  --task reacher \
  --input-h5 /data/ykz/reacher/reacher.h5 \
  --wm-policy /data/ykz/reacher/lewm_epoch_29 \
  --dry-run
```

PushT 现在默认在 `eval_profile=diffusion`、`corrective_replan` 和
`corrective_learned` 中打开 refinement；Cube/Reacher/TwoRoom 默认关闭，需要时用
`diffusion_refinement.enabled=true` 覆盖。

TwoRoom 数据如果放在 `/data/ykz/tworoom/tworoom.h5`，需要显式告诉 `eval.py`
数据位置。否则 `stable_worldmodel` 默认会去 `/home/ykz/.stable_worldmodel/tworoom.h5`
找数据，常见报错是 `FileNotFoundError`。

推荐直接指定 HDF5 文件：

```bash
./.venv/bin/python eval.py \
  --config-name tworoom \
  eval_profile=diffusion \
  +dataset_h5=/data/ykz/tworoom/tworoom.h5
```

也可以指定 cache 根目录：

```bash
./.venv/bin/python eval.py \
  --config-name tworoom \
  eval_profile=diffusion \
  cache_dir=/data/ykz/tworoom
```

`cube.yaml` 的 `profiles.diffusion` 里已经写入了完整 diffusion 配置：

```yaml
eval_profile: null
profiles:
  diffusion:
    planner_type: diffusion
    policy: /data/ykz/cube/lewm_epoch_27
    diffusion_bundle: /data/ykz/cube/diffusion_pipeline/cube_diffusion_k128_200000/diffusion_planner_best_bundle.pt
    diffusion_selection_mode: wm_only
    diffusion_num_candidates: 128
```

临时覆盖评估数量：

```bash
./.venv/bin/python eval.py --config-name cube eval_profile=diffusion eval.num_eval=10
```

测试少步推理速度：

```bash
./.venv/bin/python eval.py \
  --config-name cube \
  eval_profile=diffusion \
  diffusion_truncation_steps=1 \
  diffusion_selection_mode=wm_only \
  eval.num_eval=50
```

正式实验默认使用 `wm_only`，即用 LeWM rollout 给候选动作重排序。模型里的
score head 只为兼容旧 checkpoint 保留，后续不作为评估或优化路线。
refinement 已作为 diffusion 配置里的开关接入。除 PushT 当前 profile 默认打开外，
其他任务默认关闭；需要启用时只覆盖 `diffusion_refinement.enabled=true`：

```bash
./.venv/bin/python eval.py \
  --config-name cube \
  eval_profile=diffusion \
  diffusion_refinement.enabled=true
```

默认 refinement 参数写在每个任务配置的 `diffusion_refinement` 和
`profiles.<profile>.diffusion_refinement`：

```yaml
diffusion_refinement:
  enabled: false
  steps: 1
  step_size: 0.03
  topk: 16
  goal_weight: 1.0
  prior_weight: 0.05
  smoothness_weight: 0.005
  grad_clip_norm: 1.0
```

这里的 `topk` 是按 LeWM cost 选最低成本候选，不使用 score head。
旧的 `--task tworoom` 用法会默认映射到
`--config-name tworoom eval_profile=mpc`。

评估日志关注：

```text
[summary] success_rate=...
[planner-stats] avg_planning_time_sec=...
[planner-stats] avg_generation_time_sec=...
[planner-stats] avg_scoring_time_sec=...
```

Hydra 默认会把运行目录放到：

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

### Closed-loop Corrective Phase2

Phase2 是 training-free 的 error-triggered replanning baseline：diffusion planner
先生成 action chunk；执行中每到 LeWM latent rollout 的 checkpoint，就比较真实 latent
和预测 latent。如果 `prediction_error > corrective.error_threshold`，丢弃当前 chunk
剩余动作，并用当前观测重新调用 diffusion planner。

由于当前 LeWM rollout 按 `plan_config.action_block` 计算，PushT/TwoRoom/Cube 默认
`action_block=5`。因此 `corrective.correction_interval=2` 会实际延后到第 5、10、15...
步检查。建议 Phase2 先显式设成 `5`。

并行 eval 时必须避免 batch-global replan：某个 episode/env drift 了，只应该替换该
env 后续动作，不应该让其他 env 一起丢弃 chunk。因此 Phase2 默认使用
`trigger_scope=per_env`。`trigger_scope=batch` 只用于复现旧的全局触发行为；如果使用
batch 模式，再考虑 `trigger_stat=quantile trigger_quantile=0.9`。

PushT 推荐从阈值 sweep 开始：

```bash
./.venv/bin/python eval.py \
  --config-name pusht \
  eval_profile=corrective_replan \
  +dataset_h5=/data/ykz/pusht/pusht_expert_train.h5 \
  corrective.error_threshold=3.0
```

建议对 PushT 依次跑：

```text
corrective.error_threshold=2.5
corrective.error_threshold=3.0
corrective.error_threshold=3.5
corrective.error_threshold=4.0
corrective.error_threshold=5.0
```

TwoRoom 也可以用同样方式跑，但 Phase1 当前全成功，阈值只能先看触发频率：

```bash
./.venv/bin/python eval.py \
  --config-name tworoom \
  eval_profile=diffusion \
  +dataset_h5=/data/ykz/tworoom/tworoom.h5 \
  corrective.enabled=true \
  corrective.mode=replan \
  corrective.logging.log_prediction_error=true \
  corrective.correction_interval=5 \
  corrective.execute_horizon=25 \
  corrective.trigger_scope=per_env \
  corrective.error_threshold=1.5
```

Phase2 结果会在终端和结果文件里输出：

```text
corrective_check_count
corrective_replan_count
corrective_replan_rate
mean_prediction_error_before_replan
max_prediction_error_before_replan
prediction_error_summary
```

### Closed-loop Corrective Phase3

Phase3 是 learned corrector：不再在触发时重新调用 diffusion planner，而是用一个小
MLP 修正当前 chunk 中还没执行的动作。第一版 corrector 使用现有 planner dataset
构造 synthetic drift：给 expert prefix 和 remainder 加噪声，用 frozen LeWM rollout
得到 latent error，再监督 corrector 输出 clean expert remainder。

训练 PushT corrector：

```bash
./.venv/bin/python train_corrective_diffusion.py \
  task=pusht \
  corrective.correction_interval=5 \
  training.noise_std=0.05 \
  training.lambda_goal=0.0 \
  train.epochs=20
```

输出默认在：

```text
/data/ykz/pusht/diffusion_pipeline/pusht_corrector_ci5/
```

评估 learned corrector：

```bash
./.venv/bin/python eval.py \
  --config-name pusht \
  eval_profile=corrective_learned \
  +dataset_h5=/data/ykz/pusht/pusht_expert_train.h5
```

`pusht.yaml` 的 `profiles.corrective_learned` 已经收进了 Phase3 默认参数：

```text
corrective.enabled=true
corrective.mode=learned
corrective.corrector_path=/data/ykz/pusht/diffusion_pipeline/pusht_corrector_ci5/corrector_best_bundle.pt
corrective.correction_interval=5
corrective.execute_horizon=25
corrective.trigger_scope=per_env
corrective.error_threshold=5.0
corrective.logging.log_prediction_error=true
```

这个 profile 的作用是把原来很长的 `corrective.*` eval override 收到一个任务文件里，
避免每次手动输入一整串参数。基础 diffusion eval 使用
`--config-name pusht eval_profile=diffusion`；Phase3 learned corrector eval 使用
`--config-name pusht eval_profile=corrective_learned`。

如果要换阈值或 corrector，只 override 需要改的项：

```bash
./.venv/bin/python eval.py \
  --config-name pusht \
  eval_profile=corrective_learned \
  +dataset_h5=/data/ykz/pusht/pusht_expert_train.h5 \
  corrective.error_threshold=4.5 \
  corrective.corrector_path=/path/to/corrector_best_bundle.pt
```

Phase3 结果额外输出：

```text
corrective_correction_count
mean_correction_norm
mean_action_delta_norm
correction_time_total_sec
avg_correction_time_sec
```

当前 corrector 是固定 remainder horizon 版本。以 PushT 的 `execute_horizon=25` 和
`correction_interval=5` 为例，它训练的是 20 步 remainder，所以最适合在第一个
checkpoint 修一次；后续要支持第 10/15/20 步继续修，需要扩展成 variable-horizon
或多 corrector 版本。

`corrective.mode=learned` 会在 eval 初始化时检查 corrector 的 `remain_horizon` 是否
等于 `corrective.execute_horizon - effective_correction_interval`。如果训练 corrector
和 eval 使用的 `correction_interval/action_block/execute_horizon` 不匹配，会直接报错，
而不是静默跳过 correction。

可以先只检查 Hydra 配置是否正确组合：

```bash
./.venv/bin/python eval.py \
  --config-name pusht \
  eval_profile=corrective_learned \
  +dataset_h5=/data/ykz/pusht/pusht_expert_train.h5 \
  --cfg job
```

## Consistency Distillation

目标是把多步 diffusion planner 蒸馏成 1-2 step consistency planner，降低推理延迟。
入口：

```bash
./.venv/bin/python train_consistency_planner.py task=cube
```

配置在：

```text
config/consistency/train.yaml
config/consistency/task/cube.yaml
config/consistency/task/pusht.yaml
config/consistency/task/reacher.yaml
config/consistency/task/tworoom.yaml
```

常用命令：

```bash
./.venv/bin/python train_consistency_planner.py \
  task=cube \
  runtime.device=cuda \
  teacher.bundle_path=/path/to/diffusion_planner_best_bundle.pt \
  task.planner_dataset_path=/path/to/planner_dataset.pt \
  output.dir=/path/to/consistency_train \
  train.epochs=50 \
  distill.teacher_ode_steps=2
```

训练流程：

1. 先用 `scripts/train_diffusion_head.py` 训练完整 teacher diffusion planner，并保存 `diffusion_planner_best_bundle.pt`。
2. 用同一个 planner dataset 和 teacher bundle 启动 `train_consistency_planner.py`。
3. 训练时 student 默认从 teacher planner warm-start，但会保持可训练；frozen teacher 负责产生短 ODE bridge target，EMA student 负责 consistency target。
4. 默认损失包含 consistency matching 和 action L1 监督；`distill.goal_loss_weight>0` 时额外启用 LeWM latent goal consistency。
5. 训练完成后把 `consistency_planner_best_bundle.pt` 交给 `eval.py --config-name <task> eval_profile=consistency` 做 1-step 闭环评估。

启用 LeWM latent goal consistency：

```bash
./.venv/bin/python train_consistency_planner.py \
  task=cube \
  teacher.bundle_path=/path/to/diffusion_planner_best_bundle.pt \
  task.planner_dataset_path=/path/to/planner_dataset.pt \
  task.wm_policy=/data/ykz/cube/lewm_epoch_27 \
  distill.goal_loss_weight=0.1
```

输出：

```text
consistency_planner_best_bundle.pt
consistency_planner_last_bundle.pt
consistency_planner_ema_bundle.pt
consistency_planner_train_summary.pt
```

评估 consistency planner：

```bash
./.venv/bin/python eval.py --config-name cube eval_profile=consistency
```

每个任务配置的 `profiles.consistency` 默认使用 `diffusion_truncation_steps=1` 和
`diffusion_selection_mode=wm_only`。score head 只为 checkpoint 兼容保留，后续
consistency 实验也不再使用 `score_only` 或 `hybrid` 路线。

## 常用测试

代码级回归测试：

```bash
./.venv/bin/python -m pytest \
  tests/test_diffusion_pipeline.py \
  tests/test_diffusion_policy_refinement.py \
  tests/test_diffusion_policy_prediction_error.py \
  tests/test_prediction_error.py \
  -q
```

Consistency 相关测试：

```bash
./.venv/bin/python -m pytest tests/test_consistency_distillation.py -q
```

检查 Hydra 原始配置能否解析：

```bash
./.venv/bin/python eval.py --config-name cube eval_profile=diffusion --cfg job
./.venv/bin/python eval.py --config-name cube eval_profile=diffusion diffusion_refinement.enabled=true --cfg job
./.venv/bin/python eval.py --config-name cube eval_profile=mpc --cfg job
```

注意：`--cfg job` 是 Hydra 在进入 `run()` 之前打印的原始配置，会显示
`eval_profile` 和 `profiles`，但不会显示 `eval.py` 运行时展开后的最终 planner 字段。
真正 eval 时 `resolve_eval_profile_config()` 会把选中的 profile 合并到顶层。

检查 diffusion pipeline 配置能否解析：

```bash
./.venv/bin/python scripts/train_diffusion_head.py \
  task=tworoom \
  pipeline.device=cpu \
  pipeline.output_root=/tmp/tworoom_diffusion_dry_run \
  pipeline.dry_run=true
```

## 已做的优化

- 用 Hydra 配置统一 diffusion pipeline、eval 和 consistency distillation。
- 默认从 raw HDF5 直接构建 planner dataset，避免必须先 split 数据。
- 将 teacher planner 数据构建、action anchors、diffusion planner 训练拆成明确阶段。
- 引入 action anchors，让 diffusion planner 从典型动作块附近生成候选动作。
- `DiffusionPlannerPolicy` 支持候选动作生成后用 LeWM rollout rerank。
- 支持 `diffusion_truncation_steps` 控制推理去噪步数，便于做速度/成功率 trade-off。
- 正式评估路线固定为 `diffusion_selection_mode=wm_only`；score head 仅保留 checkpoint 兼容。
- 增加 optional world-model refinement 和 prediction-error logging；refinement top-k 使用 LeWM cost 预筛。
- 增加 consistency distillation 独立模块和 Hydra 配置，目标是 1-2 step planner。

## 原始项目

本项目基于 LeWorldModel：

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
