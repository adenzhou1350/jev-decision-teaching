# 随附模型与实验证据

[返回首页](../README.md) · [完整实验记录](../docs/experiments.md) · [重训与导出指南](../docs/getting-started.md)

这里保存课程随附的演示策略和原始报告。默认 `python -m jev_teaching` 从本目录读取文件。Qwen 底座权重不在仓库中；页面按钮模型需要额外下载与 head 指纹匹配的本地快照，或自行重训。

本目录中项目原创的合成教学数据、报告与自训演示权重按仓库的 [MIT 许可证](../LICENSE) 提供。上游模型及第三方依赖的许可不受影响。

| 目录 | 文件 | 用途 |
|---|---|---|
| `sokoban/` | [策略](sokoban/sokoban-demo.pt) | 训练前后的小型 CNN 参数，CPU 可试玩 |
| `sokoban/` | [布局](sokoban/sokoban-levels.json) / [报告](sokoban/sokoban-results.json) / [审计](sokoban/sokoban-audit.json) | 所有划分、逐局动作、汇总与回放复核 |
| `browser/` | [决策头](browser/browser-demo.pt) / [报告](browser/browser-results.json) | 冻结 Qwen 的微型共享头，逐题预测与失败 |
| `text/` | [首轮报告](text/first-run-report.json) / [运行检查](text/first-run-runtime-check.json) | 第一课 56 题的教学证据，不包含 head 或底座 |
| 根目录 | [HTTP 实测](live-check.json) | 实际服务接口的完整留出检查、延迟和返回值 |

游戏报告衡量小型合成关卡的闭环成功率，浏览器报告衡量六种合成本地页面的单步完成率。全部来自一轮固定种子实验；它们不是通用 Jev、标准大型 Sokoban 或真实网站能力排名。

自己的训练输出放在 `runs/`。推荐通过 `python -m jev_teaching.export_demo --dest runs/my-demo ...` 导出到个人目录，再用 `python -m jev_teaching --artifacts runs/my-demo` 启动，保留本目录的原始证据。命令中的 Python 应使用已安装依赖的虚拟环境，完整平台命令见安装指南。
