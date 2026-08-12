# MiniMind-O 环境配置

本项目采用以下分工：

- **Conda**：创建和激活 Python 环境，安装 FFmpeg 等系统工具。
- **uv pip**：向当前 Conda 环境安装 Python 包。
- **不使用** `uv init`、`uv add`、`uv sync`、`uv venv`。

所有命令均在 Windows PowerShell 执行。项目目录：

```powershell
Set-Location "D:\\01dpan\\01project\\life\\minimind-o"
```

## 1. 可选：当前 PowerShell 临时加速 Conda

先直接使用 Conda 默认源。只有 `conda create` 或 `conda install` 明显很慢时，才执行本节。

本节只配置 **Conda**。uv 默认下载已经很快时，不需要配置镜像。

### 1.1 当前窗口使用清华镜像

下面的命令会创建一份临时 Conda 配置，并让当前 PowerShell 窗口使用它：

```powershell
$env:CONDARC = Join-Path $env:TEMP "minimindo-condarc.yaml"
@'
channels:
  - conda-forge
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
'@ | Set-Content -Path $env:CONDARC -Encoding utf8
```

确认当前窗口已经读取临时配置：

```powershell
Write-Host $env:CONDARC
conda config --show-sources
conda config --show channels default_channels custom_channels
```

设置完成后，后面的 Conda 命令保持普通写法即可：

```powershell
conda create --name minimindo python=3.10 --yes
conda activate minimindo
conda install --channel conda-forge ffmpeg --yes
```

### 1.2 什么时候失效

`CONDARC` 是当前 PowerShell 进程的环境变量。关闭这个 PowerShell 窗口后，新窗口不会继续使用这份临时镜像配置。

如果想在当前窗口立即恢复原配置：

```powershell
Remove-Item Env:CONDARC -ErrorAction SilentlyContinue
```

临时配置文件留在系统临时目录中不会影响 Conda；需要时可以删除：

```powershell
Remove-Item (Join-Path $env:TEMP "minimindo-condarc.yaml") -ErrorAction SilentlyContinue
```

### 1.3 可选：当前窗口使用代理

如果镜像也慢，并且本机代理端口是 `7892`，可在当前 PowerShell 设置代理：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7892"
$env:HTTPS_PROXY = "http://127.0.0.1:7892"
$env:NO_PROXY = "localhost,127.0.0.1"
```

设置后重新执行刚才慢的 Conda 命令。关闭窗口后代理自动失效；也可以立即取消：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:NO_PROXY -ErrorAction SilentlyContinue
```

## 2. 重建 Conda 环境

以下命令会删除旧的 `minimindo` 环境。环境中的包和配置会被清除，但不会删除项目目录、模型权重或数据集。

```powershell
conda deactivate
conda env remove --name minimindo --yes
conda create --name minimindo python=3.10 --yes
conda activate minimindo
```

如果创建环境很慢，先执行第 1.1 节的当前窗口临时配置，然后原样重试 `conda create`。

确认当前 Python 确实来自新环境：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

预期路径类似：

```text
D:\ProgramData\miniconda3\envs\minimindo\python.exe
```

## 3. 安装 FFmpeg

FFmpeg 是可执行程序，不是 Python 包。`pydub` 播放和处理音频时需要它。

```powershell
conda install --channel conda-forge ffmpeg --yes
ffmpeg -version
```

如果这一步很慢，先执行第 1.1 节的当前窗口临时配置，然后原样重试 `conda install`。安装完成后验证：

```powershell
ffmpeg -version
```

## 4. 确认 uv 可用

uv 可以安装在系统或 Conda 的 base 环境中；它不需要安装到 `minimindo` 中。只要 PowerShell 能找到它即可：

```powershell
uv --version
Get-Command uv
```

如果提示找不到 uv，再执行：

```powershell
conda install --name base --channel conda-forge uv --yes
```

然后重新打开 PowerShell，或重新加载 Conda 初始化脚本。

## 5. 指定当前 Conda Python

后续所有 `uv pip` 命令都显式指定当前 Conda 环境的 Python，避免误装到项目 `.venv` 或其它 Python 中。

```powershell
$conda_python = Join-Path $env:CONDA_PREFIX "python.exe"
Write-Host $conda_python
```

检查 uv 看到的目标解释器：

```powershell
uv pip list --python $conda_python
```

## 6. 安装 CUDA 版 PyTorch

当前机器是 RTX 5060 Ti，优先使用 PyTorch 官方提供的 CUDA 12.8 wheel。若 PyTorch 官方安装页面给出了更新的 CUDA wheel，以官方页面命令为准，只替换下面的索引地址。

如果版本中出现 `+cpu`，说明装成了 CPU 版，需要重新安装 CUDA wheel：
uv pip uninstall --python $conda_python torch 

```powershell
#这里使用镜像，不走官方
uv pip install `
  --python $conda_python `
  torch torchvision torchaudio `
  -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

这里安装的是 PyTorch 的 CUDA runtime wheel，不等于安装系统 CUDA Toolkit。NVIDIA 驱动由系统负责，不能通过 uv 修复或替代。

## 7. 安装项目 Python 依赖

项目依赖记录在根目录的 `requirements.txt`。使用 uv 安装到同一个 Conda 环境：

```powershell
uv pip install `
  --python $conda_python `
  --requirement requirements.txt
```

如果当前网络访问清华镜像不稳定，可去掉最后一行，使用 PyPI 官方源：

```powershell
uv pip install --python $conda_python --requirement requirements.txt
```

不要运行 `uv sync`。它会按照项目的 `pyproject.toml` 和 `uv.lock` 管理项目环境，而本项目现在采用 Conda 环境加 `requirements.txt` 的方式。

## 8. 环境验证

### 8.1 检查 Python 和包位置

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch, transformers, numpy; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('numpy:', numpy.__version__)"
```

### 8.2 检查 CUDA

```powershell
python -c "import torch; print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi
```

正常情况下应看到：

- `torch.version.cuda` 不是 `None`。
- `cuda available` 为 `True`。
- `device` 显示 RTX 5060 Ti。

### 8.3 检查 FFmpeg

```powershell
Get-Command ffmpeg
ffmpeg -version
```

### 8.4 检查项目导入

```powershell
python -m compileall model dataset trainer scripts webui eval_omni.py
```

## 9. 下载模型并运行推理

模型权重放到项目的 `out` 或 `model` 目录，不要提交到 Git：

```powershell
modelscope download --model gongjy/minimind-3o-pytorch --local_dir .\out
```

运行命令行推理：

```powershell
python eval_omni.py --load_from model --weight sft_omni
```

## 10. 日常维护

激活环境后，始终使用当前 Conda Python：

```powershell
conda activate minimindo
$conda_python = Join-Path $env:CONDA_PREFIX "python.exe"
uv pip list --python $conda_python
# 示例：安装和卸载 requests
uv pip install --python $conda_python requests
uv pip uninstall --python $conda_python requests
```

导出当前环境的 Python 包清单：

```powershell
uv pip freeze --python $conda_python | Out-File requirements-local.txt -Encoding utf8
```

`requirements-local.txt` 是当前机器的完整快照，不要直接覆盖项目维护的 `requirements.txt`。

## 11. 常见错误

### `torch.cuda.is_available()` 为 `False`

先确认以下三项：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv pip show --python $conda_python torch
```

如果版本中出现 `+cpu`，说明装成了 CPU 版，需要重新安装 CUDA wheel：

```powershell
uv pip uninstall --python $conda_python torch torchvision torchaudio
uv pip install `
  --python $conda_python `
  torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu128
```

### `ModuleNotFoundError`

确认当前命令使用的是 `minimindo` 的 Python，而不是 `.venv` 或系统 Python：

```powershell
conda activate minimindo
python -c "import sys; print(sys.executable)"
uv pip list --python (Join-Path $env:CONDA_PREFIX "python.exe")
```

### 找不到 `ffmpeg.exe`

```powershell
conda activate minimindo
conda install --channel conda-forge ffmpeg --yes
Get-Command ffmpeg
```
