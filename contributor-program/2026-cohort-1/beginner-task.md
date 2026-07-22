# 新手任务：Qwen3-0.6B 单卡 GRPO × GSM8K

> 完整任务说明见 [redai-infra/Relax#76](https://github.com/redai-infra/Relax/issues/76)

## 任务目标

使用 Relax 框架，在单张 GPU 上跑完一个完整的 RL 后训练流程，完成至少 10 个训练步。

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-0.6B |
| 训练集 | GSM8K |
| 算法 | GRPO |
| 硬件 | 1× GPU |
| 预计耗时 | ~0.5 天，无需改代码 |

## 快速开始

```bash
git clone https://github.com/redai-infra/Relax.git
git clone https://github.com/redai-infra/community.git

# 脚本依赖 Relax 目录结构（自解析根目录），需 cp 到 Relax/examples/ 下再跑
mkdir -p Relax/examples/beginner-task
cp community/contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh \
   Relax/examples/beginner-task/
cd Relax

export MODEL_DIR=/your/model   # 存放 Qwen3-0.6B 的目录
export DATA_DIR=/your/data     # 存放 gsm8k/ 的目录
bash examples/beginner-task/run-qwen3-0.6B-1xgpu-grpo.sh
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

下载后即为 parquet 格式。脚本首次运行时会自动预处理（把 `answer` 字段的 CoT 剥掉，只保留 `####` 后的最终答案），生成 `main/train_clean.parquet` 并使用之，无需手动转换。

## 训练流程

```
启动脚本
  → 参数解析（Megatron-LM CLI）
  → Controller 编排服务
  → Rollout 生成（SGLang）
  → Reward 计算（math rm-type，提取 \boxed{}）
  → GRPO 优势估计 + 策略更新
  → 指标记录（ClearML）
```

## 提交内容

打包为 ZIP（以学校\_专业\_姓名命名）：

1. 完整训练日志
2. 实验报告 PDF（含 reward/loss 曲线截图、遇到的问题及解决方案）

提交地址：https://f.wps.cn/g/8L0KsIhg/
