# 新手任务：Qwen3-0.6B 单卡 GRPO × GSM8K

> 完整任务说明见 [redai-infra/Relax#76](https://github.com/redai-infra/Relax/issues/76)

## 任务目标

使用 Relax 框架，在单张 GPU 上跑完一个完整的 RL 后训练流程，完成至少 10 个训练步。

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-0.6B |
| 训练集 | GSM8K |
| 算法 | GRPO |
| 硬件 | 1× GPU（H20 80GB） |
| 预计耗时 | ~0.5 天，无需改代码 |

## 快速开始

```bash
git clone https://github.com/redai-infra/Relax.git
cd Relax

export MODEL_DIR=/your/model   # 存放 Qwen3-0.6B 的目录
export DATA_DIR=/your/data     # 存放 gsm8k/ 和 aime-2024/ 的目录
bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
```

## 准备数据

### 模型

```bash
hf download Qwen/Qwen3-0.6B --local-dir $MODEL_DIR/Qwen3-0.6B
```

### GSM8K 训练集

```bash
hf download openai/gsm8k main \
  --repo-type dataset \
  --local-dir $DATA_DIR/gsm8k
```

下载后为 parquet 格式，需转换为 JSONL：

```python
import pandas as pd, json

df = pd.read_parquet("/your/data/gsm8k/main/train-00000-of-00001.parquet")
with open("/your/data/gsm8k/train.jsonl", "w") as f:
    for _, r in df.iterrows():
        answer = r["answer"].split("####")[-1].strip()
        f.write(json.dumps({"question": r["question"], "answer": answer}) + "\n")
```

### AIME 2024 评测集

```bash
hf download AI-MO/aimo-validation-aime \
  --repo-type dataset \
  --local-dir $DATA_DIR/aime-2024
```

## 训练流程

```
启动脚本
  → 参数解析（Megatron-LM CLI）
  → Controller 编排服务
  → Rollout 生成（SGLang）
  → Reward 计算（math rm-type，提取 \boxed{}）
  → GRPO 优势估计 + 策略更新
  → 指标记录（ClearML）
  → 每 10 步在 AIME-2024 上评测
```

## 提交内容

打包为 ZIP（以学校\_专业\_姓名命名）：

1. 完整训练日志
2. 实验报告 PDF（含 reward/loss 曲线截图、遇到的问题及解决方案）

提交地址：https://f.wps.cn/g/8L0KsIhg/
