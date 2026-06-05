# Closed-loop Corrective Diffusion Planner

目标：在现有 LeWM + Diffusion Planner 上加入一个可整体消融的闭环修正模块，用 LeWM latent prediction error 检测 action chunk 执行漂移，并触发重规划或修正剩余动作。

本模块只做：

```text
Base Diffusion Planner
Error-triggered Replanning
Learned Corrector
```

不做：

```text
Hierarchical Subgoal Planner
Consistency Distillation
复杂多候选 scoring
```

## 关键修正点

1. 这不是 DCDP 的直接复现。DCDP 主要是基于执行中的观测动态特征修正 action chunk；这里改成 LeWM-specific 版本：用 `E/F` 的 latent prediction error 作为 drift signal。
2. `corrective.enabled=false` 或 `mode=none` 必须完全退化为原始 diffusion planner。
3. 先验证 prediction error 是否和失败相关，再训练 learned corrector。否则 corrector 可能只是增加复杂度。
4. learned corrector 的数据构造不能把 `u_remain` 和 `u_remain_target` 都设成同一个专家剩余动作再只用 `L_action`，这会训练成 identity，不会学会漂移修正。
5. corrector 的有效监督应优先来自“漂移后状态”的 teacher action、goal consistency rollout，或二者组合。`L_action` 更适合作为动作正则，而不是唯一目标。
6. LeWM encoder / predictor 默认 frozen；goal consistency 可以让梯度流到 corrected action，但不能更新 LeWM 参数。

## 统一配置

```yaml
corrective:
  enabled: false
  mode: "none"        # none, replan, learned
  correction_interval: 2
  execute_horizon: 4
  error_threshold: 0.5
  error_metric: "l2"  # l2, mse, cosine

  corrector_model:
    hidden_dim: 512
    num_layers: 3
    dropout: 0.1
    predict_residual: true
    residual_scale: 1.0

  training:
    enabled: false
    perturb_action_std: 0.05
    perturb_action_prob: 0.5
    lambda_action: 1.0
    lambda_goal: 1.0
    lambda_smooth: 0.0

  logging:
    log_prediction_error: true
    log_correction_norm: true
    log_replan_count: true
```

核心公式：

```text
z_start = E(o_t)
z_goal  = E(o_g)
u       = D(z_start, z_goal)

z_pred = F(z_start, u[:k])
z_real = E(o_{t+k})
e      = z_real - z_pred
err    = distance(z_real, z_pred)
```

## Phase 1: Prediction Error Logging

目标：只记录 drift signal，不改变策略行为。

需要做：

1. 新增 `compute_prediction_error(z_real, z_pred, metric)`。
2. 支持 latent `[B, D]`；如果是 token latent `[B, N, D]`，先对 token 维 mean pooling。
3. 支持：

```text
l2     = ||z_real - z_pred||_2
mse    = mean((z_real - z_pred)^2)
cosine = 1 - cosine_similarity(z_real, z_pred)
```

4. 在 eval loop 里每执行 `correction_interval` 步记录一次：

```text
prediction_error
prediction_error_mean
prediction_error_max
prediction_error_at_correction
success
final_goal_error
```

5. `mode=none` 时也允许 logging，但不得改变 action。
6. 验证：失败 episode 的 prediction error 是否显著更高；如果没有相关性，先不要进入 learned corrector。

建议文件：

```text
utils/prediction_error.py
utils/action_smoothness.py
tests/test_prediction_error.py
evaluation/eval_corrective_diffusion.py 或现有 eval loop
```

验收：

```text
corrective.enabled=false 行为不变
metric 输出 shape 为 [B]
encoder/predictor 无梯度更新
eval 能输出 prediction_error 统计
```

## Phase 2: Error-triggered Replanning

目标：实现 training-free closed-loop baseline。

流程：

```text
1. z_start = E(obs_t), z_goal = E(goal_obs)
2. u = D(z_start, z_goal)
3. 执行 u[:correction_interval]
4. z_real = E(obs_after_k)
5. z_pred = F(z_start, u[:correction_interval])
6. err = compute_prediction_error(z_real, z_pred)
7. 如果 err > error_threshold：
      丢弃剩余动作，从 z_real 重新调用 D
   否则：
      继续执行 u[correction_interval:execute_horizon]
```

需要做：

1. `mode=replan` 只改 eval loop，不训练新网络。
2. `error_threshold`、`correction_interval`、`execute_horizon` 全部走配置。
3. `execute_horizon=1` 时打印 warning：每步重规划本身已经接近 closed-loop，收益可能很小。
4. 记录：

```text
replan_count
replan_rate
mean_prediction_error_before_replan
ms_per_plan
total_eval_time
```

验收：

```text
mode=replan 不加载 corrector
err <= threshold 时继续剩余 chunk
err > threshold 时重新规划
能和 Base Diffusion Planner 输出同格式结果
```

## Phase 3: Learned Corrector

目标：训练轻量网络修正剩余 action chunk，而不是每次重新规划。

模型：

```python
class ActionChunkCorrector(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        max_remain_horizon: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        predict_residual: bool = True,
        residual_scale: float = 1.0,
    ):
        ...
```

输入输出：

```text
input  = [z_real, z_goal, error_latent, flatten(u_remain)]
output = delta_u_remain 或 u_remain_corr
shape  = [B, R, action_dim]
```

推理：

```text
1. u = D(z_start, z_goal)
2. 执行 u[:correction_interval]
3. z_real = E(obs_after_k)
4. z_pred = F(z_start, u[:correction_interval])
5. e = z_real - z_pred
6. u_remain = u[correction_interval:execute_horizon]
7. u_corr = corrector(z_real, z_goal, e, u_remain)
8. 执行 u_corr
```

训练数据修正：

```text
不要：u_remain_target = u_remain 且只用 L_action
原因：这会学 identity，不会学 correction
```

推荐第一版监督：

```text
z_tilde = F(z_start, noisy_prefix)
z_pred  = F(z_start, clean_prefix)
e       = z_tilde - z_pred
u_base  = diffusion_planner(z_tilde, z_goal)[:R] 或 clean plan remainder
target  = teacher/CEM/expert action from z_tilde if available
z_target = E(obs[t+H]) 或 z_goal
```

损失：

```text
L = lambda_action * L_action
  + lambda_goal   * L_goal
  + lambda_smooth * L_smooth

L_action = |u_corr - u_target|_1
L_goal   = ||F(z_real, u_corr) - z_target||_2^2
L_smooth = mean(||u_corr[i+1] - u_corr[i]||_2^2)
```

如果暂时没有可靠 `u_target`，先使用 `L_goal + L_smooth`，把 `L_action` 作为靠近 base action 的正则：

```text
L_action = |u_corr - u_base|_1
```

需要做：

1. 新增 `ActionChunkCorrector` 和 shape tests。
2. 新增 corrector dataset，构造 action-noise drift 样本。
3. 新增 corrector training script；只训练 corrector，冻结 `E/F/D`。
4. 加载 corrector 到 `mode=learned` eval。
5. 记录：

```text
prediction_error
correction_norm
action_delta_norm
ms_per_correction
```

建议文件：

```text
models/action_chunk_corrector.py
datasets/corrector_dataset.py
training/train_corrector.py
tests/test_action_chunk_corrector.py
```

验收：

```text
batch size 1 和 >1 都通过
remain horizon 固定为 execute_horizon - correction_interval
corrector loss 能下降
learned 模式只在 corrective.mode=learned 时启用
```

## Phase 4: Unified Evaluation

目标：输出三种方法的统一对比，先小任务跑通，再扩展四任务。

第一轮实验：

```text
Tasks:
- Reacher
- Push-T

Methods:
- Base Diffusion Planner
- Error-triggered Replanning
- Learned Corrector
```

核心指标：

```text
success_rate
final_goal_error
episode_length
prediction_error_mean
prediction_error_max
replan_count
replan_rate
correction_norm_mean
action_smoothness
action_jerk
ms_per_plan
ms_per_correction
total_eval_time
```

可选扩展：

```text
correction_interval: 1, 2, 4
error_threshold: 0.25, 0.5, 1.0
action_noise_std: 0.0, 0.05, 0.1
Tasks: Two-Room, Reacher, Push-T, OGBench-Cube
```

判断标准：

```text
prediction_error 和失败相关：
  LeWM error 可以作为 drift detector

replan 提升成功率：
  training-free closed-loop baseline 有效

learned corrector 接近或超过 replan 且更快：
  corrector 有论文价值

execute_horizon=1 没有明显提升：
  正常，因为每步重规划已经接近 closed-loop
```

最终表格：

```text
method | task | success_rate | final_goal_error | pred_err_mean | action_jerk | ms_plan | ms_correction | replan_count
```

最终验收：

```text
1. Base 模式完全不变。
2. Replan 模式不训练新网络。
3. Learned 模式能输出修正后的剩余 action chunk。
4. corrector loss 能下降。
5. E/F 默认 frozen。
6. eval 输出 prediction_error、correction_norm、replan_count、action_jerk、ms_per_correction。
7. Push-T 和 Reacher 先跑通完整 evaluation。
8. 四个任务能输出统一结果表。
```
