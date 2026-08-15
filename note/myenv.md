# MiniMind-O 环境配置

本项目采用以下分工：

- **Conda**：创建和激活 Python 环境，安装 FFmpeg 等系统工具。
- **uv pip**：向当前 Conda 环境安装 Python 包。
- **不使用** `uv init`、`uv add`、`uv sync`、`uv venv`。

所有命令均在 Windows PowerShell 执行。项目目录：

```powershell
Set-Location "D:\\01dpan\\01project\\life\\minimind-o"
同意条款：
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

### 1 可选：当前窗口使用代理
如果镜像也慢，并且本机代理端口是 `7892`，可在当前 PowerShell 设置代理：
$env:HTTP_PROXY = "http://127.0.0.1:7892"
$env:HTTPS_PROXY = "http://127.0.0.1:7892"
$env:NO_PROXY = "localhost,127.0.0.1"
设置后重新执行刚才慢的 Conda 命令。关闭窗口后代理自动失效；也可以立即取消：
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:NO_PROXY -ErrorAction SilentlyContinue

## 2. 重建 Conda 环境
以下命令会删除旧的 `minimindo` 环境。环境中的包和配置会被清除，但不会删除项目目录、模型权重或数据集。
conda deactivate
conda env remove --name minimindo --yes
conda create --name minimindo python=3.10 --yes
conda activate minimindo

# 确认当前 Python 确实来自新环境：
python --version
python -c "import sys; print(sys.executable)"
预期路径类似：D:\ProgramData\miniconda3\envs\minimindo\python.exe

## 3. 安装 FFmpeg，FFmpeg 是可执行程序，不是 Python 包。`pydub` 播放和处理音频时需要它。
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
conda clean -i --yes

conda install -c conda-forge ffmpeg --yes

#这里有gdk报错：
conda deactivate
conda env remove -n minimindo -y
chcp 65001
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
conda create -n minimindo -c conda-forge --override-channels python=3.10 ffmpeg -y

  这句话的含义是：
  - -n minimindo：创建环境名叫 minimindo
  - -c conda-forge：使用 conda-forge 仓库
  - --override-channels：只用我指定的仓库，不要再混入 defaults
  - python=3.10 ffmpeg：一开始就把 Python 和 ffmpeg 一起装进去

conda activate minimindo
ffmpeg -version

## 4. 确认 uv 可用
uv 可以安装在系统或 Conda 的 base 环境中；它不需要安装到 `minimindo` 中。只要 PowerShell 能找到它即可：
uv --version
Get-Command uv
如果提示找不到 uv，再执行：
conda install --name base --channel conda-forge uv --yes
然后重新打开 PowerShell，或重新加载 Conda 初始化脚本。

## 5. 指定当前 Conda Python
后续所有 `uv pip` 命令都显式指定当前 Conda 环境的 Python，避免误装到项目 `.venv` 或其它 Python 中。
$conda_python = Join-Path $env:CONDA_PREFIX "python.exe"
Write-Host $conda_python

检查 uv 看到的目标解释器：
uv pip list --python $conda_python

## 6. 安装 CUDA 版 PyTorch
当前机器是 RTX 5060 Ti，优先使用 PyTorch 官方提供的 CUDA 12.8 wheel。若 PyTorch 官方安装页面给出了更新的 CUDA wheel，以官方页面命令为准，只替换下面的索引地址。
如果版本中出现 `+cpu`，说明装成了 CPU 版，需要重新安装 CUDA wheel：
uv pip uninstall --python $conda_python torch 
#这里使用镜像，不走官方.大约下载需要半小时
$conda_python = "D:\ProgramData\miniconda3\envs\minimindo\python.exe"
uv pip uninstall --python $conda_python torch torchvision torchaudio -y
uv pip install `
  --python $conda_python `
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 `
  -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ `
  -i https://mirrors.aliyun.com/pypi/simple/ `
  --trusted-host mirrors.aliyun.com

不用uv：
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 `
>>     -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ `
>>     -i https://mirrors.aliyun.com/pypi/simple/ `
>>     --trusted-host mirrors.aliyun.com
！！！这里还是手动吧~ 自己找torch的配套轮子文件：
https://download.pytorch.org/whl/torch_stable.html

验证 GPU 版：
uv run --python $conda_python python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

## 7. 安装项目 Python 依赖
项目依赖记录在根目录的 `requirements.txt`。使用 uv 安装到同一个 Conda 环境：

uv pip install `
  --python $conda_python `
  --requirement requirements.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

不要运行 `uv sync`。它会按照项目的 `pyproject.toml` 和 `uv.lock` 管理项目环境，而本项目现在采用 Conda 环境加 `requirements.txt` 的方式。
```powershell
## 8. 环境验证
### 8.1 检查 Python 和包位置
python -c "import sys; print(sys.executable)"
python -c "import torch, transformers, numpy; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('numpy:', numpy.__version__)"

### 8.2 检查 CUDA
python -c "import torch; print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi
```

正常情况下应看到：

- `torch.version.cuda` 不是 `None`。
- `cuda available` 为 `True`。
- `device` 显示 RTX 5060 Ti。
```powershell

### 8.3 检查 FFmpeg
Get-Command ffmpeg
ffmpeg -version

### 8.4 检查项目导入
python -m compileall model dataset trainer scripts webui eval_omni.py

## 9. 下载模型并运行推理
模型权重放到项目的 `out` 或 `model` 目录，不要提交到 Git：
modelscope download --model gongjy/minimind-3o-pytorch --local_dir .\out
运行命令行推理问答：
python eval_omni.py --load_from model --weight sft_omni

## 10. 日常维护
激活环境后，始终使用当前 Conda Python：
conda activate minimindo
$conda_python = Join-Path $env:CONDA_PREFIX "python.exe"
uv pip list --python $conda_python
# 示例：安装和卸载 requests
uv pip install --python $conda_python requests
uv pip uninstall --python $conda_python requests

导出当前环境的 Python 包清单：
uv pip freeze --python $conda_python | Out-File requirements-local.txt -Encoding utf8
```
`requirements-local.txt` 是当前机器的完整快照，不要直接覆盖项目维护的 `requirements.txt`。
