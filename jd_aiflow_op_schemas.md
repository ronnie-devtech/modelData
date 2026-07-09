# jd.aiflow 算子 Schema 文档

---

## jd.aiflow.MoE

### 算子简介

**Mixture of Experts（混合专家）** 算子，支持任意 FC hidden size 的 MoE 前向推理。

算子接受一个输入 token 序列，通过 router 概率向量将每个 token 路由给对应的专家网络。每个专家网络由 1～3 层全连接层（FC1、FC2、FC3）构成，每层可选偏置和激活函数。每个 router_probs 输入对应一个独立的输出，支持多个 router 同时驱动一次前向，输出结果一一对应。

支持 `float32` 和 `float16` 两种精度。

---

### 属性（Attributes）

| 属性名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `activation` | STRING | `"relu"` | 激活函数，可选：`relu`、`gelu`、`silu`、`identity`。支持以逗号分隔为每层分别指定，如 `"relu,gelu"` |
| `expert_num` | INT | `1` | 每次从专家池中选取的 top-k 专家数 |
| `input_prob_num` | INT | `1` | 输入 router_probs 的数量（即并发路由数，对应输出数量） |
| `layer_num` | INT | `3` | MoE 内部的 FC 层数（1～3） |

---

### 输入（Inputs）

输入序列布局为：`x, w0, b0, w1, b1, w2, b2, prob0, prob1, ..., probn`

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `input` | 必选 | T | `(num_rows, input_size)` 或 `(batch_size, seq_len, input_size)` | 输入 token 特征，支持 2D 或 3D |
| 1 | `fc1_experts_weights` | 必选 | T | `(input_size, expert_num, fc1_n)` | 第一层专家权重，**注意维度顺序与常规 3D 不同** |
| 2 | `fc1_experts_bias` | 可选 | T | `(num_experts, fc1_n)` | 第一层专家偏置 |
| 3 | `fc2_experts_weights` | 必选 | T | `(num_experts, fc1_n, fc2_n)` | 第二层专家权重 |
| 4 | `fc2_experts_bias` | 可选 | T | `(num_experts, fc2_n)` | 第二层专家偏置 |
| 5 | `fc3_experts_weights` | 可选 | T | `(num_experts, fc2_n, fc3_n)` | 第三层专家权重 |
| 6 | `fc3_experts_bias` | 可选 | T | `(num_experts, fc3_n)` | 第三层专家偏置 |
| 7 | `router_probs0` | 必选 | T | `(num_rows, num_experts)` | 第 0 个 router 的专家概率 |
| 8 | `router_probs1` | 可选 | T | `(num_rows, num_experts)` | 第 1 个 router 的专家概率 |
| 9 | `router_probs2` | 可选 | T | `(num_rows, num_experts)` | 第 2 个 router 的专家概率 |
| 10–22 | `router_probs3` ~ `router_probs15` | 可选 | T | `(num_rows, num_experts)` | 第 3～15 个 router 的专家概率，共支持最多 **16** 个 router |

> **注意**：索引 20 的 `router_probs12` 与索引 19 同名（源码中的命名 bug），实际为第 13 个 router（`router_probs13`）；索引 21 为 `router_probs14`，索引 22 为 `router_probs15`。

---

### 输出（Outputs）

每个 `router_probs{i}` 对应一个输出 `output{i}`，输出数量与有效 router_probs 输入数量一致（最多 16 个）。

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `output0` | 必选 | T | `(num_rows, fc3_n/fc2_n)` 或 `(batch_size, seq_len, fc3_n/fc2_n)` | router_probs0 对应的 MoE 输出 |
| 1 | `output1` | 可选 | T | 同上 | router_probs1 对应的 MoE 输出 |
| 2–15 | `output2` ~ `output15` | 可选 | T | 同上 | 对应各 router 的 MoE 输出 |

输出的最后一维为最后一个有效 FC 层权重的输出维度（`fc3_n` 若有第三层，否则 `fc2_n`）。

---

### 类型约束（Type Constraints）

| 类型变量 | 约束类型 | 说明 |
|---|---|---|
| `T` | `tensor(float)`, `tensor(float16)` | 所有输入输出及权重均为此类型 |

---

### 计算逻辑说明

```
对每个 router_probs[i]:
  1. 使用 router_probs[i] 中的 top-expert_num 概率，为每个 token 选取专家
  2. 将 token 分发给对应专家，执行 FC1 → act → [FC2 → act → [FC3]] 前向
  3. 将各专家输出按概率加权聚合，得到 output[i]
```

多个 router_probs 共享同一套专家权重（`fc*_experts_weights`），一次前向调用输出多路结果，节省重复计算。

---

### 使用示例

```
输入:
  input:               [1024, 512]         (1024 tokens, input_size=512)
  fc1_experts_weights: [512, 8, 2048]      (input_size=512, expert_num=8, fc1_n=2048)
  fc1_experts_bias:    [8, 2048]
  fc2_experts_weights: [8, 2048, 512]      (expert_num=8, fc1_n=2048, fc2_n=512)
  fc2_experts_bias:    [8, 512]
  router_probs0:       [1024, 8]           (num_rows=1024, num_experts=8)
  router_probs1:       [1024, 8]

属性:
  activation: "relu"
  expert_num: 2         (top-2 routing)
  input_prob_num: 2
  layer_num: 2

输出:
  output0: [1024, 512]
  output1: [1024, 512]
```

---

## jd.aiflow.StackGemm

### 算子简介

**堆叠多专家 Gemm** 算子，对多个专家并行执行最多 4 层全连接（FC1～FC4），可选前置 BatchNormalization，输出所有专家的结果堆叠在一起。与 MoE 不同，StackGemm 不做路由选择，而是对输入同时过所有专家，适用于需要保留全部专家输出的场景。

支持 `float32` 和 `float16` 两种精度。

---

### 属性（Attributes）

| 属性名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `activation` | STRING | `"relu"` | 激活函数，可选：`relu`、`gelu`、`silu`、`identity`。支持逗号分隔为每层分别指定 |
| `expert_num` | INT | `1` | 专家数量 |
| `layer_num` | INT | `3` | FC 层数（1～4） |
| `standard_mode` | INT | `1` | 输出维度排列模式：`0` 为 `(expert_num, batch_size, hidden)`，非 `0` 为 `(batch_size, expert_num, hidden)` |
| `single_expert_mode` | INT | `1` | 单专家时的输出形状模式：`0` 与多专家相同；`1` 去掉 expert 维；`2` 仅对 3D 输入有效，保持前两维与输入一致 |

---

### 输入（Inputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `input` | 必选 | T | `(num_rows, input_size)` 或 `(batch_size, seq_len, input_size)` | 输入特征，支持 2D 或 3D |
| 1 | `gamma` | 可选 | T | `(num_experts, input_size)` | BatchNormalization gamma 参数 |
| 2 | `beta` | 可选 | T | `(num_experts, input_size)` | BatchNormalization beta 参数 |
| 3 | `fc1_experts_weights` | 必选 | T | `(num_experts, input_size, fc1_n)` | 第一层专家权重 |
| 4 | `fc1_experts_bias` | 可选 | T | `(num_experts, fc1_n)` | 第一层专家偏置 |
| 5 | `fc2_experts_weights` | 必选 | T | `(num_experts, fc1_n, fc2_n)` | 第二层专家权重 |
| 6 | `fc2_experts_bias` | 可选 | T | `(num_experts, fc2_n)` | 第二层专家偏置 |
| 7 | `fc3_experts_weights` | 可选 | T | `(num_experts, fc2_n, fc3_n)` | 第三层专家权重 |
| 8 | `fc3_experts_bias` | 可选 | T | `(num_experts, fc3_n)` | 第三层专家偏置 |
| 9 | `fc4_experts_weights` | 可选 | T | `(num_experts, fc3_n, fc4_n)` | 第四层专家权重 |
| 10 | `fc4_experts_bias` | 可选 | T | `(num_experts, fc4_n)` | 第四层专家偏置 |

---

### 输出（Outputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `output` | 必选 | T | 见下 | 所有专家的输出堆叠结果 |

输出 shape 取决于 `standard_mode` 和 `single_expert_mode`：

- 多专家，`standard_mode != 0`：`(batch_size, expert_num, last_hidden)`
- 多专家，`standard_mode == 0`：`(expert_num, batch_size, last_hidden)`
- 单专家，`single_expert_mode == 1`：`(batch_size, last_hidden)` 或 `(batch_size, seq_len, last_hidden)`
- 单专家，`single_expert_mode == 2`（仅 3D 输入）：`(batch_size, seq_len, last_hidden)`

其中 `last_hidden` 为最后一个有效 FC 层的输出维度。

---

### 类型约束（Type Constraints）

| 类型变量 | 约束类型 | 说明 |
|---|---|---|
| `T` | `tensor(float)`, `tensor(float16)` | 所有输入输出及权重均为此类型 |

---

### 计算逻辑说明

```
1. 可选：对 input 执行 BatchNormalization（使用 gamma、beta）
2. 对每个专家并行执行：FC1 → act → [FC2 → act → [FC3 → act → [FC4]]]
3. 将所有专家的输出按 standard_mode 堆叠为输出张量
```

---

### 使用示例

```
输入:
  input:               [256, 128]          (256 tokens, input_size=128)
  fc1_experts_weights: [4, 128, 512]       (4 experts, input_size=128, fc1_n=512)
  fc2_experts_weights: [4, 512, 128]       (fc1_n=512, fc2_n=128)

属性:
  activation:        "relu"
  expert_num:        4
  layer_num:         2
  standard_mode:     1
  single_expert_mode: 1

输出:
  output: [256, 4, 128]   (batch_size=256, expert_num=4, fc2_n=128)
```

---

## jd.aiflow.GroupedBatchedGemm

### 算子简介

**分组批量 Gemm** 算子，针对 `split → matmul+add → concat` 融合模式设计。将输入沿行方向按 `size_splits` 分成若干段，每段与对应的权重矩阵相乘并加偏置，最后将结果拼接输出。

支持 `float32` 和 `float16` 两种精度。

---

### 属性（Attributes）

无。

---

### 输入（Inputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `left_X` | 必选 | T | `(s0+s1+…+sn, K)` 或 `(1, s0+s1+…+sn, K)` | 输入矩阵，行方向按 size_splits 分段 |
| 1 | `right_X` | 必选 | T | `(n, K, N)` | 每组对应的权重矩阵，共 n 组 |
| 2 | `Bias` | 可选 | T | `(n, N)` | 每组对应的偏置 |
| 3 | `size_splits` | 必选 | M | `(n,)`，值为 `[s0, s1, …, sn]` | 每组的行数 |

---

### 输出（Outputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `output` | 必选 | T | `(s0+s1+…+sn, N)` 或 `(1, s0+s1+…+sn, N)` | 各组 Gemm 结果拼接后的输出 |

---

### 类型约束（Type Constraints）

| 类型变量 | 约束类型 | 说明 |
|---|---|---|
| `T` | `tensor(float)`, `tensor(float16)` | 输入输出及权重类型 |
| `M` | `tensor(int64)` | size_splits 类型 |

---

### 计算逻辑说明

```
将 left_X 按 size_splits 切分为 [X_0, X_1, ..., X_n]
对每组 i：Y_i = X_i @ right_X[i] + Bias[i]
output = concat([Y_0, Y_1, ..., Y_n], axis=0)
```

---

### 使用示例

```
输入:
  left_X:      [300, 64]         (s0=100, s1=200, K=64)
  right_X:     [2, 64, 32]       (n=2 组, K=64, N=32)
  Bias:        [2, 32]
  size_splits: [100, 200]

输出:
  output: [300, 32]
```

---

## jd.aiflow.SplitSequenceMoE

### 算子简介

**分段序列 MoE** 算子，针对 `split → matmul+add+act+… → add+LayerNorm → concat` 融合模式设计。将输入按 `size_splits` 切分后，每段分别经过两层 FC（W1、W2）和 SkipLayerNormalization，最后将各段结果拼接输出。

支持 `float32` 和 `float16` 两种精度。

---

### 属性（Attributes）

| 属性名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `activation` | STRING | `"Relu"` | 第一层 FC 后的激活函数，可选：`Relu`、`Gelu`、`Identity` |
| `epsilon` | FLOAT | `9.999e-9` | SkipLayerNormalization 的 epsilon 值 |

---

### 输入（Inputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `X` | 必选 | T | `(s0+s1+…+sn, K)` | 输入矩阵，行方向按 size_splits 分段 |
| 1 | `W1` | 必选 | T | `(n, K, N)` | 每组第一层权重 |
| 2 | `Bias1` | 可选 | T | `(n, N)` | 每组第一层偏置 |
| 3 | `W2` | 必选 | T | `(n, N, K)` | 每组第二层权重（输出维度回到 K） |
| 4 | `Bias2` | 可选 | T | `(n, K)` | 每组第二层偏置 |
| 5 | `Gamma` | 必选 | T | `(n, K)` | SkipLayerNorm 的 gamma 参数 |
| 6 | `Beta` | 必选 | T | `(n, K)` | SkipLayerNorm 的 beta 参数 |
| 7 | `size_splits` | 必选 | M | `(n,)`，值为 `[s0, s1, …, sn]` | 每组的行数 |

---

### 输出（Outputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `output` | 必选 | T | `(s0+s1+…+sn, K)` | 各组输出拼接结果，最后维度与输入 K 相同 |

---

### 类型约束（Type Constraints）

| 类型变量 | 约束类型 | 说明 |
|---|---|---|
| `T` | `tensor(float)`, `tensor(float16)` | 输入输出及权重类型 |
| `M` | `tensor(int64)` | size_splits 类型 |

---

### 计算逻辑说明

```
将 X 按 size_splits 切分为 [X_0, X_1, ..., X_n]
对每组 i：
  H_i = act(X_i @ W1[i] + Bias1[i])
  Y_i = SkipLayerNorm(H_i @ W2[i] + Bias2[i] + X_i, Gamma[i], Beta[i])
output = concat([Y_0, Y_1, ..., Y_n], axis=0)
```

---

### 使用示例

```
输入:
  X:           [300, 64]         (s0=100, s1=200, K=64)
  W1:          [2, 64, 256]      (n=2 组, K=64, N=256)
  W2:          [2, 256, 64]      (N=256, K=64)
  Gamma:       [2, 64]
  Beta:        [2, 64]
  size_splits: [100, 200]

属性:
  activation: "Relu"
  epsilon:    1e-9

输出:
  output: [300, 64]
```

---

## jd.aiflow.RecRankCalibration

### 算子简介

**推荐排序校准（Rec Rank Calibration）** 算子，用于对排序模型的原始预测分（`task_scores`）进行校准，输出经过分桶映射后的 logit 值。

算法原理：将 `task_scores` 映射到预先计算好的 `bucket_size` 个桶上，对每个样本计算加权累积校准分：

```
cali = SUM(ReLU(e_i + w_i) * v_i) / bucket_size，i in [0, k]
output = log(cali / (1 - cali))
```

其中 `k` 为 `task_scores * bucket_size` 的整数部分，`v_i` 为小数部分权重。输出结果被 clip 到 `[epsilon, 1]` 范围后再取 logit。

支持 pointwise（2D）和 listwise（3D）两种输入模式。

---

### 属性（Attributes）

| 属性名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bucket_size` | INT | `100` | 分桶数量，必须与 `task_relu` 最后一维一致 |
| `epsilon` | FLOAT | `1e-16` | 数值稳定性 epsilon，用于 clip 和校验 |

---

### 输入（Inputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `task_relu` | 必选 | T（double） | `(batch_size, bucket_size)` 或 `(list_size, batch_size, bucket_size)` | 每个样本的分桶权重，已经过 ReLU(e + w) |
| 1 | `task_scores` | 必选 | T（double） | `(batch_size, 1)` 或 `(list_size, batch_size, 1)` | 排序模型原始预测分，范围 `[0, 1]`，不可为 NaN |

---

### 输出（Outputs）

| 索引 | 名称 | 是否必选 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|---|
| 0 | `output` | 必选 | R（float） | `(batch_size, 1)` 或 `(list_size, batch_size, 1)` | 校准后的 logit 分数 |

---

### 类型约束（Type Constraints）

| 类型变量 | 约束类型 | 说明 |
|---|---|---|
| `T` | `tensor(double)` | 输入类型 |
| `R` | `tensor(float)` | 输出类型 |

---

### 使用示例

```
输入:
  task_relu:   [512, 100]    (batch_size=512, bucket_size=100)
  task_scores: [512, 1]      (每个样本的原始预测分，范围 [0,1])

属性:
  bucket_size: 100
  epsilon:     1e-16

输出:
  output: [512, 1]           (float32 logit 校准分)
```

```
# listwise 模式
输入:
  task_relu:   [10, 512, 100]   (list_size=10, batch_size=512, bucket_size=100)
  task_scores: [10, 512, 1]

输出:
  output: [10, 512, 1]
```
