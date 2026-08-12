# MiniMind-O 精读资源

## Knowledge

- [MiniMind-O 项目 README](../README.md)
  项目作者提供的环境、架构、推理和训练总览。用于确认项目约定和复现步骤。
- [MiniMind-O 推理入口](../eval_omni.py)
  本项目命令行推理的第一手源码。用于追踪模型初始化和各模态推理入口。
- [MiniMind-O 模型实现](../model/model_omni.py)
  Thinker、Talker、两个 Projector 和生成循环的第一手源码。
- [MiniMind-O Technical Report](https://arxiv.org/abs/2605.03937)
  项目作者发布的技术报告。用于校验模型设计目标、结构和实验结论。
- [PyTorch: `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
  PyTorch 官方权重反序列化说明。用于理解检查点如何加载和映射到设备。
- [PyTorch: `nn.Module.load_state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
  PyTorch 官方状态字典加载说明。用于理解 `strict=False` 和权重键匹配。
- [Hugging Face: Chat templates](https://huggingface.co/docs/transformers/main/chat_templating)
  Transformers 官方对话模板说明。用于理解消息如何变成模型实际读取的 token 序列。

## Wisdom (Communities)

- [MiniMind-O GitHub Issues](https://github.com/jingyaogong/minimind-o/issues)
  项目用户和维护者的问题记录。用于交叉验证复现问题和实际运行经验。

## Gaps

- RTX 4060 Laptop 8GB 上完整训练复现的可靠参数尚未验证，进入训练阶段后通过最小实验补齐。
