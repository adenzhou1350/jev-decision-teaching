# 首轮验证：真实 4060、小数据、仅训练 head

[返回首页](../README.md) · [第一课](lessons/01-text.md) · [安装指南](getting-started.md)

日期：2026-10-09，Windows 原生 Python；没有启动 WSL，没有远端训练或业务动作。

## 固定设置

- 本地模型：Qwen3.5-0.8B，原有已下载快照；完整文件 SHA256 见报告。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8GB。
- Python 3.14、PyTorch 2.11.0+cu128、Transformers 5.14.1。
- BF16 冻结 backbone，SDPA，DeltaNet 使用 PyTorch fallback；head 在 CPU 上以 FP32 训练。
- 56 条原创手写题，224 个状态/候选对；24 条参与梯度，8 条用于检查点选择。
- 训练 200 步，学习率 0.01，随机种子 42；dev NLL 选择第 200 步。
- backbone 可训练参数：0；共享 head 可训练参数：1024。
- 数据 SHA256：`982f8ca1edb61078c9bc301b9e594bb88e9a6a23de15c2d395b2a100144e6b84`。

## 首轮结果

| Split | 随机 head | 训练后 head |
| --- | ---: | ---: |
| train | 4/24 | 24/24 |
| dev | 1/8 | 7/8 |
| test | 1/12 | 11/12 |
| challenge | 3/12 | 10/12 |

每个 split 四类均衡，均匀随机期望和常量多数类准确率均为 25%。
随机 head 不是原模型的生成式问答基线。

留出错例完整保留：

| 输入 | 期望 | 实际 |
| --- | --- | --- |
| 解除暂停，往后执行 | continue | cancel |
| 如果我说暂停，你会怎么做 | other | pause |
| 算了……我还没决定 | other | cancel |

这说明学会常见表达和理解引用/假设不是一回事。
test 与 challenge 合计为 21/24，但不能把合计当生产流量准确率。
本例为教学开发样本，作者可见且可修改，不是独立盲测或模型能力排名。

## 耗时和显存的正确读法

- 冻结特征提取：19.2624 秒，覆盖全部 56 题，模型加载时间不计入。
- 200 步 head 优化及 dev 选择：0.1533 秒；不是完整模型训练或端到端耗时。
- PyTorch 峰值 allocated：1711.80 MiB；不是系统总 GPU 占用，不是 LoRA 训练需求。
- 没有做与生成式 LLM 的等质量速度比较，也没有测常驻 HTTP 服务延迟。

入库证据：[`first-run-report.json`](../artifacts/text/first-run-report.json)。此副本只隐去本机模型绝对路径，
预测、指标与文件 SHA256 均未改变。原始运行输出保留在本地，不会被后续运行覆盖。
backbone 与 `head.pt` 不入库；读者可以重新训练获得自己的 head。

## 验证范围

6 项单元测试通过：数据契约、跨 split group 泄漏、答案不进 prompt、
概率及候选置换、变长候选推理、人工可分特征训练和 head 重载。
人工特征测试只证明优化实现，不用于上述语言准确率。

真实模型重载与重新前向检查由 [jev_teaching/verify_runtime.py](../jev_teaching/verify_runtime.py) 单独记录。
复跑时先按第一课训练自己的 head，再从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m jev_teaching.verify_runtime --out runs/text-mine --device cuda
```

Linux/macOS 将 Python 路径换成 `.venv/bin/python`。历史首轮的 head 不随仓库分发；自行训练得到的是新的实验，不能要求其数字与旧报告完全相同。
其范围仅限两条样例，不能推广为所有 batch/精度/硬件都一致。

实际检查结果：同样 batch=4 重载后，两条样例的概率差为 0；
逆序候选并改为 batch=1、重新做 backbone 前向后，两条 top-1 都未改变，
但最大概率变化分别为 **0.125099** 和 **0.043887**（约 12.51、4.39 个百分点）。
两项改变同时发生，不能将差异单独归因于 BF16、padding 或候选顺序。
它不是小到可以忽略的概率差，不宣称概率级运行一致或置信度已校准。
记录见 [`first-run-runtime-check.json`](../artifacts/text/first-run-runtime-check.json)。该副本只修正 scope 的措辞：
原来的 BF16 batch drift 字样不是因果诊断，数值记录全部保留。
