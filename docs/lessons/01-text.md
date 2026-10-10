# 最简 Jev-style 入门：让小模型选择，不让它写答案

[返回首页](../../README.md) · [安装指南](../getting-started.md) · [第二课：推箱子](02-sokoban.md)

目标：读懂并跑通一次 **候选打分 → 训练 → 留出评测 → 保存 → 再调用**。
这是独立教学实现，不是 TypeSafe Jev 的源码、训练方法复现或能力替代品。
没有调用 Jev API，没有 RL，没有代理框架，也不执行任何真实任务控制动作。

## 1. 先理解这一个公式

给定用户状态 `s`、问题 `q` 和候选描述 `c_i`：

```text
h_i = FrozenBackbone(s, q, c_i) 的最后一个有效 token 的隐藏向量
z_i = w · LayerNorm(h_i)
p_i = softmax(z)_i
loss = -log(p_正确候选)
```

`w` 是所有候选共用的一个打分向量，不是“四个动作各有一个输出神经元”。
候选 ID 只用于返回结果，模型看到的是候选的自然语言描述。
所以输入新候选不需要改输出层形状；但**形状支持新候选，不等于语义泛化已经成立**。

与普通聊天模型的区别：这里不调用 `generate()`，不逐 token 写答案，不用解析 JSON。
输入仍然需要经过 Transformer，绝不是“没有 LLM 计算”。
四个候选作为四条条件序列批量编码，重复计算状态；**第一版没有共享前缀加速**。

## 2. 这个例子具体做什么？

把中文请求分为暂停、取消、继续、其他。例如：

```json
{
  "state": "不要取消，只是暂停",
  "question": "根据用户当前明确意图选择动作",
  "candidates": {
    "pause": "暂时暂停当前任务，保留进度，之后可以恢复",
    "cancel": "取消并结束当前任务，不再执行",
    "continue": "继续或恢复执行当前任务",
    "other": "没有明确的任务控制要求，不执行控制动作"
  },
  "target": "pause"
}
```

`target` 只参与训练损失，不进入模型输入。问题和完整规则以 [jev_teaching/data.py](../../jev_teaching/data.py) 为准。
只有 56 条原创手写数据：24 train、8 dev、12 test、12 challenge。
challenge 专门保留否定、引用、反悔等情况。它们不是自然分布，也不是大规模 benchmark。
表达之间仍可能语义相近；没有相同文本、group 跨 split，不代表消除了所有语义泄漏。

## 3. 安装与训练

先按 [安装指南第 2 节](../getting-started.md#2-启用文本或页面按钮模型) 创建虚拟环境、安装 `requirements-models.txt`，并下载模型快照。
以下命令从仓库根目录执行，使用 Windows 虚拟环境完整路径；Linux/macOS 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`，其余参数相同，无需激活环境。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_mini_jev.py -v
```

GPU 版本的 PyTorch 请按 [官方安装选择器](https://pytorch.org/get-started/locally/) 匹配驱动安装。
依赖文件是兼容范围，不是完整锁文件。实际验证版本另记在结果文件中。

先冻结模型，再用 `no_grad()` 提取候选特征，最后在 CPU 上训练一个很小的 FP32 线性头。
这样不用保存大模型的反向传播激活，也不用训练 0.8B 个参数。

```powershell
# 使用完整本地快照；有 NVIDIA GPU 且 PyTorch 支持 CUDA 时使用 cuda
.\.venv\Scripts\python.exe -X utf8 -m jev_teaching.mini_jev train --model models/Qwen3.5-0.8B --local-only --device cuda --out runs/text-mine

# 没有 GPU 可改为 --device cpu；CPU 的速度和精度路径需要单独验证
```

若不指定 `--model`，公开模型默认 `Qwen/Qwen3.5-0.8B`；不加 `--local-only` 时可从远端下载。远端下载建议额外指定 `--revision <完整commit>` 固定版本；
本地运行会对配置、tokenizer 和权重文件计算 SHA256，写进报告和 head 检查点。
没有完整权重的缓存目录不能作为有效模型，程序不会用随机权重替代加载失败。
输出目录已存在时拒绝覆盖，复跑请换 `--out`。

Qwen3.5 在缺少可选 DeltaNet 加速依赖时会走 PyTorch fallback。这不妨碍教学，
但此时不能拿它的速度宣传专门推理引擎的性能。

## 4. 训练到底是哪几行？

阅读 [jev_teaching/mini_jev.py](../../jev_teaching/mini_jev.py)，先找 `fit_head()`：

```python
logits = head(x).squeeze(-1)        # [题数, 候选数]
loss = F.cross_entropy(logits, y)  # 直接传 logits，别提前 softmax
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

其余代码负责准备特征、保存证据和避免静默错误。
训练最多 200 步，只按 dev NLL 选择检查点；test/challenge 不参与梯度或检查点选择。
只保存 head，不重复保存完整 backbone。**这是线性探针式教学，不含 LoRA，也不是全参数微调。**

推荐阅读顺序：`data.py` → `prompt()` → `Encoder.encode()` → `fit_head()` → `metrics()`。
本版训练函数为简洁起见要求每题候选数一致；推理/指标代码支持不同数量的候选。

## 5. 看结果与调用

```powershell
.\.venv\Scripts\python.exe -X utf8 -m jev_teaching.mini_jev predict --out runs/text-mine --device cuda --local-only --text "不要取消，只是暂停"
```

预测从检查点恢复 backbone 路径和 revision；本地模型文件若改变会拒绝加载旧 head。
每次命令行调用都会重新加载 backbone；这不是常驻服务延迟。
目前只返回候选与概率，不连接机器人、业务 API 或其他执行器。

`runs/<名字>/report.json` 包含：

- 未训练随机 head 和训练后 head 的逐条预测。
- 均匀随机的期望准确率、常量多数类基线。
- 四个 split 的正确数、准确率、NLL 和 Brier score。
- 数据 SHA256、模型文件 SHA256、版本、训练参数数目和显存峰值。
- 特征提取耗时、只训练 head 的耗时，二者分开记录。

**随机 head 不是原始 Qwen 的生成式能力基线。** 结果不能解读成“训练前 Qwen 只有这么准”。
head 的置换测试复用已经提取的特征，证明共享读出不依赖候选位置；
它不是 GPU 换 batch、换精度、重做 backbone 前向的一致性证明。

首轮真实运行的摘要见 [文本验证记录](../text-validation.md)。不预设高准确率目标，不删错例。
真实前向检查还观察到换 batch/padding 后的明显概率变化，细节也在验证记录中；
数学上的候选置换性质和数值实现中的运行一致性必须分开。

## 6. 为什么这还不能直接上线？

1. 24 条训练数据过小，且动作描述固定，很可能学会捷径。
2. softmax 必须在候选里选一个；`other` 是一个已训练动作，不是自动 OOD 检测器。
3. 概率还没有校准。0.99 不等于真实正确率 99%。
4. 不理解任务当前是否可暂停，也没验证执行成功；决策与执行是两件事。
5. 这里没有与生成式模型做等任务、等硬件、等质量的端到端速度比较。

下一课可选：扩充真人表达 → 独立校准集 → 低置信回退 → LoRA → 真实任务轨迹。
不要一次全加进来。小例子效果一般也有教学价值：可以具体定位是特征不够，还是数据不够。

## 7. 与已有项目的关系

- [NanoJev](https://github.com/TianyuCodings/NanoJev)：完整决策训练和游戏闭环；本例不复制其游戏成绩。
- [Open-Jev](https://github.com/Zefan-Cai/Open-Jev)：LoRA、共享标量头和校准；本例只讲最小读出。
- [Valen](https://github.com/Liuziyu77/Valen)：视觉决策、训练和评测，适合后续进阶。
- [JevAny 训练指南](https://simplejev.org/JevAny/docs/TRAINING.html)：现成的入门训练参考。
- [Qwen3.5 官方文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)：backbone 与 fallback 注意事项。

上述项目是参考与后续阅读，不主张本例首次提出候选打分或复现原版 Jev。

本课的语义控制例子请自行训练；第二、三课另附各自的演示策略和决策头，均不包含 Qwen 底座权重。
