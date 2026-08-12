# Mission: 精读并复现 MiniMind-O

## Why

通过完整阅读和复现 MiniMind-O，掌握端到端多模态模型的代码组织、模型架构、数据流和训练流程，为后续独立修改模型结构打下基础。

## Success looks like

- 能独立运行并验证文本、音频和图像推理
- 能解释 Thinker、Talker、Audio Projector 和 Vision Projector 的职责与数据流
- 能沿代码说明主要张量的形状变化
- 能解释数据集、损失函数、反向传播和权重保存流程
- 能在 RTX 4060 Laptop 8GB 环境中完成可行的训练复现实验
- 能提出、实现并验证第一个模型结构修改

## Constraints

- 每天约投入 4 小时
- 使用 RTX 4060 Laptop 8GB
- 以代码结构和模型架构为主
- 原理推导以理解代码所需为限
- 目前有一定 LLM 基础，但没有完整手写模型代码的经验

## Out of scope

- 第一阶段不修改模型结构
- 第一阶段不深入完整 WebUI
- 暂不研究多机多卡训练
- 暂不进行完整数学证明
