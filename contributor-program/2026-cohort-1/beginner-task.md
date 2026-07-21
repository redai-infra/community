# 新手任务：Qwen3-0.6B 单卡 GRPO × GSM8K

目标：使用 Relax 框架跑通一次完整的强化学习后训练流程，完成不少于 10 个 training step，并提交实验报告与训练曲线。

本任务面向初次使用 Relax 的同学，重点是走通主链路、理解框架结构、形成可复现记录，完成本次任务可以领取小红书"薯队长"玩偶。

完成后，你应能说明这条主链路：

启动脚本 → 参数解析 → Controller 编排 → Rollout 生成 → Reward 计算 → GRPO 更新 → 指标记录

---

## 一、基本要求

| 项目 | 要求 |
| --- | --- |
| 代码仓库 | redai-infra/Relax（以活动指定分支/commit 为准） |
| 训练算法 | GRPO（`--advantage-estimator grpo`） |
| 模型 | Qwen3-0.6B |
| 数据集 | GSM8k |
| 计算资源 | 单卡 |
| 训练步数 | 不少于 10 step |
| 预计工作量 | 约 0.5 天，不要求修改代码 |

---

## 二、数据与模型下载

### 模型

```bash
hf download Qwen/Qwen3-0.6B --local-dir /your/model/Qwen3-0.6B
```

### 训练集

```bash
hf download openai/gsm8k main \
  --repo-type dataset \
  --local-dir /your/data/gsm8k
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

### 测试集

```bash
hf download AI-MO/aimo-validation-aime \
  --repo-type dataset \
  --local-dir /your/data/aime-2024
```

国内可配置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 三、快速启动

```bash
git clone https://github.com/redai-infra/Relax.git
cd Relax
export MODEL_DIR=/your/model
export DATA_DIR=/your/data
bash examples/beginner-task/run-qwen3-0.6B-1xgpu-grpo.sh
```

镜像参考：[Relax Docker Image List · redai-infra/Relax · Discussion #30](https://github.com/redai-infra/Relax/discussions/30)

---

## 四、提交材料

**📬 提交地址：** https://f.wps.cn/g/8L0KsIhg/

### 1、训练日志

（附上训练过程日志文件）

### 2、实验报告（PDF 格式）

压缩包命名：`北京大学+计算机科学与技术+张三.zip`

#### 1. 基本信息

| 项目 | 填写内容 |
| --- | --- |
| 学校 / 专业 / 姓名 | \<填写\> |
| 完成日期 | \<YYYY-MM-DD\> |
| 代码分支 / commit id | \<填写\> |
| GPU 型号与数量 | \<填写\> |
| Python / CUDA / 镜像版本 | \<填写\> |

#### 2. 任务详情

| 项目 | 填写内容 |
| --- | --- |
| 模型 | Qwen3-0.6B |
| 数据集 | GSM8K |
| 算法 | GRPO |
| 实际训练步数 | \<填写\> |
| 总耗时 | \<填写\> |
| 日志 / 输出目录 | \<填写\> |
| 主要参数调整 | \<未调整，或写明参数和值\> |

完整启动命令：

```bash
# 粘贴本次实际使用的完整命令
```

#### 3. 实验结果曲线

## 3.1 Reward 曲线

插入 reward_curve.png，并注明指标名、数据来源

## 3.2 Loss 曲线

插入 loss_curve.png，并注明指标名、数据来源

## 3.3 其他曲线（可选）

插入 KL、grad norm、response length 等曲线

#### 4. 遇到的问题与解决方案

| 问题现象 / 报错 | 原因分析 | 解决方案 | 如何确认已解决 |
| --- | --- | --- | --- |
| \<填写\> | \<填写\> | \<填写\> | \<填写\> |

> 如未遇到明显报错，请记录一个实际排查过的风险点，例如显存不足、模型或数据路径错误、Ray 启动失败、reward 未写入日志。

#### 5. 总结

用 3～5 句话说明：任务是否完成、最终 step、对训练主链路的理解，以及如果继续训练会优先调整什么。
