# 第二课：让共享打分器真正玩完推箱子

[返回首页](../../README.md) · [安装指南](../getting-started.md) · [第三课：页面按钮](03-browser.md)

这节课训练一个小型棋盘网络，展示从数据到环境交互的完整链路，无需语言模型权重。

## 任务与观察

棋盘为 6×6，边界是墙，内部最多 3 个随机墙格。单箱任务用于训练/dev/test，双箱任务作为 OOD。
策略看到四个二值平面：墙、目标、箱子、玩家。动作是上、右、下、左。
走一步或推一步都算一个环境步；箱子只能推，不能拉；全部箱子落在目标上即通关。

[jev_teaching/sokoban_env.py](../../jev_teaching/sokoban_env.py) 的 `step()` 是唯一规则实现，碰墙或推不动会保持原状态。
训练和演示均用规则屏蔽当前不可执行的动作；这一步不会检测未来死锁。

## 用求解器做老师

`ExactOracle` 枚举某个小布局的所有合法玩家/箱子状态，将转移边反向，从所有已通关状态做 BFS。
因此得到每个可解状态的精确最短距离，以及所有最优下一步。
若两个方向都能走最短路径，就都作为正确动作，训练目标均分到这些动作。

求解器只负责离线生成标签。在线 `rollout()` 与服务 `/api/step` 只用当前棋盘和学到的权重。
[tests/test_sokoban_training.py](../../tests/test_sokoban_training.py) 会禁用 oracle 并改动标签，检查动作是否仍然不变。

## 防止记住测试地图

按“墙壁 + 目标位置”定义布局组，进一步把 90 度旋转和镜像等价布局合并。
同组所有玩家/箱子起点必须在同一个 split。
训练含距离目标 1 步的收尾状态；评测起点的最短距离至少 4 步、最多 30 步。

首轮固定数据：256 个训练布局、11,151 个训练状态；dev/test 各 32 个布局、128 个起点；
双箱 OOD 为 12 个布局、48 个起点。生成种子 20261009，参数初始化种子 42。

## 模型只有两部分

```python
h = encoder(board_planes)                 # 每个状态只编码一次
z_i = shared_scorer(concat(h, action_i))   # 四个候选共用同一个打分器
probabilities = softmax(mask_illegal(z))
loss = -(optimal_action_distribution * log_softmax(z)).sum()
```

编码器是三层卷积与一个全连接层，候选是方向坐标；整个模型 360,097 参数。
训练是监督行为克隆，含旋转/镜像增强；没有强化学习奖励，也没有在线搜索。
共享打分器不依赖候选顺序，但本课只训练了四种移动，不能据此推断会使用任意新技能。

## 训练与选择

先完成首页的 CPU 试玩安装。所有命令从仓库根目录执行；Linux/macOS 把 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`。以下先用 CPU，有可用 CUDA 时可改成 `--device cuda`；已记录的 GPU 训练时间不代表 CPU 速度。

```powershell
.\.venv\Scripts\python.exe -m jev_teaching.sokoban_train --device cpu --epochs 60 --out runs/sokoban-mine
.\.venv\Scripts\python.exe -m jev_teaching.audit_results --run runs/sokoban-mine
.\.venv\Scripts\python.exe -m jev_teaching.export_demo --sokoban-run runs/sokoban-mine --dest runs/my-demo
.\.venv\Scripts\python.exe -m jev_teaching --artifacts runs/my-demo
```

最多 60 个 epoch，每 5 个 epoch 用 dev 闭环通关率选择检查点，通关率相同时看 dev NLL。
首轮选中第 55 个 epoch；test/OOD 不参与选择。
导出结果放入个人目录 `runs/my-demo/sokoban/`，不会改动随附的 `artifacts/sokoban/`；复跑时换新的运行/导出目录名。

## 必须评测整局

每个策略从同一起点独立运行，最多 96 步；每次行动后读取新棋盘。
确定性策略一旦回到已访问状态，即可判定会重复循环并提前结束；随机策略允许继续到步数上限。
训练前、训练后、随机合法动作、贪心距离规则均使用相同合法动作掩码。

| 策略 | 单箱 test | 双箱 OOD |
| --- | ---: | ---: |
| 同架构随机初始化 | 2/128 | 0/48 |
| 随机合法动作 | 28/128 | 4/48 |
| 贪心曼哈顿距离 | 14/128 | 2/48 |
| 训练后 | 122/128 | 25/48 |

单箱成功的 122 局平均 6.57 步，均达到对应起点的最短步数；仍有 6 局失败。
双箱下降到 52.1%，显示它没有学成通用规划器。
这是一轮小型程序生成实验：有些起点只需少量推动，成功率不能外推到标准 Boxoban 或大型复杂关卡。

网页可以选任意 test/OOD 起点，手动接管会明确标记，避免将人工辅助记为模型独立通关。
逐局动作见 [sokoban-results.json](../../artifacts/sokoban/sokoban-results.json)，接口复测见 [live-check.json](../../artifacts/live-check.json)。

## 可以自己改的练习

1. 保持测试集不动，增加训练布局，观察通关率而不只看单步正确率。
2. 在新实验里加入双箱训练，再用新的未见双箱布局评测。
3. 比较四输出分类头与共享候选头；本课不能单独证明共享头比分类头优越。
4. 增加棋盘尺寸，需要重新设计网络输入和明确新的任务划分。

如根据本轮 test 错例改模型，该测试集就已经成为开发材料；最终结论需要另外冻结新测试集。
