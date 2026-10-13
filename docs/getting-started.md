# 安装、试玩与重训

[返回首页](../README.md) · [文本决策头](lessons/01-text.md) · [推箱子](lessons/02-sokoban.md) · [页面按钮](lessons/03-browser.md)

所有命令均从 clone 的仓库根目录执行。需要 Python 3.11+；实测版本为原生 Windows Python 3.14.3。Linux/macOS 命令采用对应平台路径，尚未逐平台验证。不需要 `pip install -e .`，也不需要激活虚拟环境。

## 1. 只想先玩推箱子


Windows PowerShell：

```powershell
git clone https://github.com/adenzhou1350/jev-decision-teaching.git
cd jev-decision-teaching
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jev_teaching
```

Linux / macOS：

```bash
git clone https://github.com/adenzhou1350/jev-decision-teaching.git
cd jev-decision-teaching
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jev_teaching
```

打开 [本地页面](http://127.0.0.1:8765)。选择关卡后可切换策略、单步、自动运行或手动接管。此时只装 PyTorch，不下载语言模型；浏览器语义模型尚未启用。关闭服务时在终端按 Ctrl+C。

基础命令的完整名称是 `python -m jev_teaching.serve_demo`，与 `python -m jev_teaching` 等价。默认读取 `artifacts/` 中随附的演示文件。

## 2. 启用文本或页面按钮模型

这两课需要额外依赖和完整 Qwen3.5-0.8B 快照。下载前可查看 [官方模型页](https://huggingface.co/Qwen/Qwen3.5-0.8B)；模型底座不包含在本仓库。下面下载的快照用于训练自己的 head，不直接交给随附的页面 head。

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\hf.exe download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

Linux / macOS：

```bash
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

下载完成后，页面课程接着做[训练 head → 导出 → 启动个人演示目录](#页面按钮给自己的快照训练-head)；文本课程按[第一课](lessons/01-text.md)训练和调用。训练与启动要使用同一份本地快照。

后面的训练和服务命令中，没有 NVIDIA CUDA 可用 `--device cpu`；CPU 语言模型性能未作对照测量。需要 GPU 时按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为同一个 `.venv` 安装匹配版本，不要只在系统 Python 中安装。

随附页面 head 的[报告](../artifacts/browser/browser-results.json)记录了底座文件指纹，公开记录未给出对应的 Hugging Face revision。默认下载会取得当时的模型快照，不能据此保证与随附 head 匹配；已有匹配快照时，可走下方的[单独启动路径](#已有匹配快照运行随附页面-head)。

若虚拟环境没有 `hf` 可执行文件，可在该环境安装或更新 `huggingface_hub`，或按官方模型页说明下载完整快照。不要只下载配置文件；模型目录需要 tokenizer、配置及完整权重。

## 3. 自己训练并导出到独立目录

`runs/` 存放自己的实验；训练命令拒绝覆盖已有运行目录。下例使用新的个人演示目录，不改动仓库自带的 `artifacts/` 结果。重复实验时换一个输出名称。

### 推箱子：无需预训练模型

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m jev_teaching.sokoban_train --device cpu --out runs/sokoban-mine
.\.venv\Scripts\python.exe -m jev_teaching.audit_results --run runs/sokoban-mine
.\.venv\Scripts\python.exe -m jev_teaching.export_demo --sokoban-run runs/sokoban-mine --dest runs/my-demo
.\.venv\Scripts\python.exe -m jev_teaching --artifacts runs/my-demo
```

Linux / macOS：

```bash
.venv/bin/python -m jev_teaching.sokoban_train --device cpu --out runs/sokoban-mine
.venv/bin/python -m jev_teaching.audit_results --run runs/sokoban-mine
.venv/bin/python -m jev_teaching.export_demo --sokoban-run runs/sokoban-mine --dest runs/my-demo
.venv/bin/python -m jev_teaching --artifacts runs/my-demo
```

有可用 CUDA 时可以把训练的 `--device cpu` 改为 `cuda`。已记录的训练耗时来自 RTX 4060，不代表 CPU 速度。

### 页面按钮：给自己的快照训练 head

先按第 2 节安装模型依赖并下载快照。`--include-bundled-sokoban` 明确把随附的推箱子模型和数据一起导出，让个人演示目录的两个页签都能使用，无需再训练推箱子。

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m jev_teaching.browser_train --model models/Qwen3.5-0.8B --local-only --device cuda --out runs/browser-mine
.\.venv\Scripts\python.exe -m jev_teaching.export_demo --browser-run runs/browser-mine --include-bundled-sokoban --dest runs/browser-demo
.\.venv\Scripts\python.exe -m jev_teaching --artifacts runs/browser-demo --model models/Qwen3.5-0.8B --device cuda
```

Linux / macOS：

```bash
.venv/bin/python -m jev_teaching.browser_train --model models/Qwen3.5-0.8B --local-only --device cuda --out runs/browser-mine
.venv/bin/python -m jev_teaching.export_demo --browser-run runs/browser-mine --include-bundled-sokoban --dest runs/browser-demo
.venv/bin/python -m jev_teaching --artifacts runs/browser-demo --model models/Qwen3.5-0.8B --device cuda
```

导出后文件分别放在 `<dest>/sokoban/` 和 `<dest>/browser/`。已有自己的推箱子 run 时，可直接同时传入 `--sokoban-run` 和 `--browser-run`，无需 `--include-bundled-sokoban`。导出工具会拒绝覆盖已有文件；只有明确要替换自己选定的结果时才加 `--overwrite`。

更小的文本决策头训练与单条预测见 [第一课](lessons/01-text.md)，其输出不直接作为网页浏览器模型使用。

### 已有匹配快照：运行随附页面 head

仅在本地快照与[随附报告中的文件指纹](../artifacts/browser/browser-results.json)逐项匹配时使用。先停掉已运行的服务，再从仓库根目录启动：

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m jev_teaching --model models/Qwen3.5-0.8B --device cuda
```

Linux / macOS：

```bash
.venv/bin/python -m jev_teaching --model models/Qwen3.5-0.8B --device cuda
```

这条路径使用随附的 `artifacts/`，服务首次运行页面模型时核验底座文件并加载。若指纹不匹配，回到上面的个人 head 训练流程，不绕过校验。随附页面模型的测试完成率为 70.8%，概率未经校准；这个数字不代表你重训的 head 会得到相同结果。

## 4. 复核而不训练

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m jev_teaching.audit_results --artifacts artifacts/sokoban --out runs/my-audit.json
# 先启动服务；另开一个位于仓库根目录的终端
.\.venv\Scripts\python.exe -m jev_teaching.verify_demo --out runs/my-http-check.json
```

Linux / macOS：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m jev_teaching.audit_results --artifacts artifacts/sokoban --out runs/my-audit.json
# 先启动服务；另开一个位于仓库根目录的终端
.venv/bin/python -m jev_teaching.verify_demo --out runs/my-http-check.json
```

HTTP 检查读取已启动的服务；启用了浏览器模型才会验证对应页面。输出文件不能已存在。原始证据的入口见 [artifacts/README.md](../artifacts/README.md)。

## 常见问题

- **`No module named jev_teaching`：**回到 clone 根目录运行模块命令，不要进入 `jev_teaching/` 再执行。
- **`No module named torch/transformers`：**用文档中的完整虚拟环境 Python 路径安装和运行；语言模型课需要 `requirements-models.txt`。
- **CUDA 不可用：**先用 CPU 完成推箱子试玩，再核对该虚拟环境的 PyTorch 是否支持 CUDA；无需启动本地 WSL。
- **模型身份不匹配：**对自己的本地快照重训 head，导出到新的 `runs/...` 目录；不要绕过指纹检查。
- **端口 8765 被占用：**启动时加 `--port 8766`，浏览器打开对应端口；HTTP 检查也要加 `--url http://127.0.0.1:8766`。
- **输出目录或报告已存在：**换新名称；保留旧实验便于比较。`--overwrite` 仅是导出工具的显式选项。

旧版的 `python serve_demo.py` 等根目录命令已迁移为 `python -m jev_teaching` 或 `python -m jev_teaching.<工具名>`；随附模型与历史报告保持不变。
