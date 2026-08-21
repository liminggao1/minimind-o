# MiniMind-O 技术报告：一种开放的小规模原生语音全模态模型

作者：Jingyao Gong（独立研究者）  
邮箱：gongjy.cs@foxmail.com  
项目：[GitHub](https://github.com/jingyaogong/minimind-o) · [Hugging Face](https://huggingface.co/collections/jingyaogong/minimind-o) · [ModelScope](https://modelscope.cn/collections/gongjy/minimind-o)  
技术报告，arXiv:2605.03937v1，2026 年 5 月 5 日

## 摘要

MiniMind-O 是建立在 MiniMind 语言模型之上的开放式、约 0.1B 参数规模的全模态模型。它接收文本、语音和图像输入，同时输出文本和流式语音。项目发布了模型代码、检查点，以及 T2A（文本到音频）、I2T（图像到文本）和 A2A（音频到音频）的主要 Parquet 训练数据，因此完整交互链路可以直接检查。

模型使用完整 MiniMind 主干作为 Thinker，并使用由 MiniMind 模块构成的独立四层 Talker。冻结的 SenseVoice-Small 和 SigLIP2 编码器提供语音与图像特征；轻量 MLP 投影器将这些特征映射到 MiniMind 隐藏空间，再注入对应的模态占位符位置。Talker 读取 Thinker 的中间层状态，以及自回归的八层 Mimi-code 缓冲区。说话人控制通过专用 speaker token、右对齐的参考 codec 提示和预计算的 192 维 CAM++ 嵌入实现，因此声音条件属于音频 code 上下文，而不是独立的 TTS 模块。

在 Talker 隐藏维度为 768 时，稠密版和 MoE 版在 Thinker–Talker 一致性评估中的平均字符错误率（CER）分别为 0.0897 和 0.0900，整体声音克隆相似度分别为 0.5995 和 0.5937。除了报告一个可运行的系统，本文还指出小型全模态模型中三个对规模关键的设计选择：中间层语义桥接、公开的多模态序列格式，以及参数高效的八码本接口。

> 注释：CER 越低表示 Talker 生成的语音越接近 Thinker 生成的文本；CAM++ 相似度是说话人嵌入余弦相似度，越高表示音色越接近参考说话人。

## 1 引言

GPT-4o、Qwen-Omni、Moshi 以及近期的语音-文本系统，使实时多模态交互从产品接口问题变成了模型设计问题。一个可用的系统必须能够听、看、推理、说话，并在用户打断时停止说话。通常的工程路线仍然是级联：ASR 将语音转成文本，LLM 写出回答，TTS 将回答渲染为波形。这条路线有效，但语言模型处在声学回路之外。一旦语音模块是外部模块，发音、时序和说话人控制中的错误就很难归因到共享表示。

MiniMind-O 从相反的约束出发。基础模型是 MiniMind，而不是十亿级主干，因此每个新增模态都必须通过一个很小的隐藏空间。这使系统成为全模态模型设计的压力测试：在大规模下只是便利的组件，在 0.1B 规模下必须被明确表示并能够测量。图 1 的设计保持语义路径和声学路径分离。Thinker 就是 MiniMind Transformer 本身。它接收普通文本嵌入，以及在音频和图像占位符位置注入的 SenseVoice 与 SigLIP2 投影状态。Talker 是一个独立的四层模块；当存在兼容权重时，它从 MiniMind 模块初始化。这样，语义预测保留在语言主干中，音频 code 生成拥有自己的循环历史。

**图 1：MiniMind-O 架构。** 音频和图像由冻结的 SenseVoice 与 SigLIP2 编码器处理，经 MLP 投影器映射到 MiniMind 隐藏空间后，注入模态占位符位置。Thinker 的中间层状态与 Mimi-code 历史融合后送入独立 Talker，由 Talker 预测八层 codec，用于流式语音生成。

模型规模不仅是约束，也是主要实验变量。MiniMind-O 的目标是实现一个小型且完全可检查的全模态系统：在激活参数规模约 0.1B 的情况下，同时支持文本、语音和图像输入以及流式语音输出。在这个规模上，桥接、投影器和 codec 接口必须确实必要、可测量、可复现。

第二个结果涉及架构。八个 Mimi codebook 原本可以各自使用独立的嵌入表和输出头。实际实验表明，共享基座加非满秩 adapter 的参数化方式具有清晰的参数效率曲线：中等秩已经能够恢复大部分收敛和 codebook 准确率收益；解耦秩实验还显示，输出头的秩比输入嵌入的秩更重要。因此，低秩接口是有实验支持的设计选择，而不是单纯的实现捷径。

第三个要点是桥接层。如果 Talker 读取最终的下一个 token 预测状态，它会继承当前文本 token 以及语言模型输出头几何结构的强偏置；这对文本 logits 有用，却会成为嘈杂的声学条件。如果读取过浅的状态，模型又还没有积累足够上下文来解决发音、句法或跨模态指代。一个简单的普通话例子是字符“地”（U+5730），其读音会随上下文变化。原始 embedding 不包含这种上下文相关的读音信息，而中间隐藏状态可以携带足够的周围信息，同时又没有完全坍缩为下一个 token 分类器。

发布内容的第四部分是数据集。如果代码公开，但对齐数据、codec 目标和模态布局是隐含的，全模态系统就很难复现。因此 MiniMind-O 同时发布主要 T2A、I2T 和 A2A Parquet 数据集，以及消费这些数据的代码路径。该数据集并非最终的通用语料，而是这套小模型方案的训练基底，其中包含文本、图像字节、语音输入、Mimi code 目标、参考 code 提示和说话人嵌入，格式可检查、可修改。

**图 2：Talker 侧语音生成设计。** Talker 接收 Thinker 桥接状态、音频 code embedding、可选的说话人信息以及参考 codec 提示，输出八层 Mimi codebook logits，再解码为波形。


发布的系统有两个变体：稠密版 `minimind-3o` 和 `minimind-3o-moe`，二者的激活参数规模大致相同。音频输入由 SenseVoice-Small 编码，图像输入由 SigLIP2 编码；语音输出使用八个 Mimi codebook 表示，并解码为 24 kHz 音频。说话人条件由两种与规模相匹配的信号注入：参考 codec 提示和 192 维 CAM++ 说话人嵌入。由于模型前向传播过程中不会调用说话人编码器，这种选择也使推理路径保持可检查。

因此，这条语音路径更接近上下文内条件控制，而不是固定说话人的 TTS 头。默认发布版本提供五个内置语音提示：`dylan`、`eric`、`serena`、`uncle_fu` 和 `vivian`；另有七个语音提示作为评测中的留出提示。推理时，更换声音只会改变右对齐的参考 Mimi code，以及放置在 `<|audio_spk|>` 位置的 CAM++ 向量。Thinker 提示和 Talker 权重保持不变，因此音色迁移是共享音频 code 布局的属性，而不是另一条微调路径。

报告还记录了在这一小规模设置中被确定为重要的设计因素：从何处提取 Thinker 状态、Talker 需要多宽、参考语音应如何放入音频缓冲区、发布的数据如何组织，以及哪些评测能够揭示内容不匹配而不只是音频质量。这些细节并非无关紧要的实现选择。在 0.1B 规模下，桥接位置、可复用数据和参数高效的 codebook 接口会直接影响完整链路能否保持可训练、可复现。因此，本文的贡献不是一个新的大型模型，而是一套紧凑、可检查的方案，将原生语音全模态交互转化为可控的研究对象。

## 2 相关工作

### 全模态与语音-文本对话模型

GPT-4o 使原生语音的多模态交互受到广泛关注；随后 Qwen2.5-Omni 和 Qwen3-Omni 让 Thinker–Talker 方案更加具体：可以从语义路径提取隐藏状态，再由流式运行的语音路径消费这些状态。开放系统探索了相近的选择。Mini-Omni 展示了语言模型仍在生成文本时进行语音流式输出；Mini-Omni2 加入了视觉和双工交互。LLaMA-Omni、VITA、GLM-4-Voice、Baichuan-Audio、Step-Audio 和 Spirit-LM 研究了语音交互、音频语言理解以及交错的口语-书面语建模等组合。

MiniMind-O 以这些工作为参照，研究一个互补问题：当激活参数规模被压到约 0.1B 时，哪些组件仍然必要？哪些接口选择能让完整链路可复现，而不只是做出一个演示？

**图 3：Thinker 与 Talker 的训练序列格式。** 文本监督作用于 Thinker 的回答 token，音频监督作用于目标 Mimi code 位置；参考 code 区域只作为条件上下文，不作为损失目标。

### 离散音频表示与语音生成

离散音频 token 使 Talker 可以用语言模型式目标进行训练。VALL-E 表明 codec token 足以支持零样本 TTS；MusicGen 将多码本自回归变成标准生成模式；EnCodec 和 SNAC 提供了实用的神经 codec。Moshi 在语音-文本系统中引入了流式音频 codec Mimi，MOSS-Audio-Tokenizer 则研究面向未来音频基础模型的可扩展 tokenizer 设计。

MiniMind-O 保留 Mimi 的八码本表示。区别在于预测器所在位置：音频 code 预测器挂接在一个很小的全模态模型上，而不是交给大型独立声学模型。

### 多模态特征对齐

在视觉语言建模中，CLIP 和 BLIP-2 确立了感知与语言建模的实用分工：冻结或缓慢更新的编码器生成特征，桥接模块将特征映射到 LLM 空间。LLaVA、Qwen-VL、Qwen2-VL 和 SigLIP2 在此基础上提供了更强的视觉表示和指令微调能力。MiniMind 系列在纯语言和视觉语言版本中也采用了相同的极简方案理念。

在当前 MiniMind-O 代码库中，音频和视觉都使用普通的两层 MLP 投影器。这是更简单的选择：外部编码器负责感知，投影器只需把隐藏状态映射到 MiniMind embedding 空间。

## 3 模型架构

图 1 展示了 `model_omni.py` 实现的数据路径。文本进入原生 token embedding 表。语音转换为 SenseVoice 前端特征，再经过冻结的 SenseVoice 编码器；输出状态由 `MMAudioProjector` 映射，该投影器是带 LayerNorm 和 GELU 的两层 MLP。图像由冻结的 SigLIP2 视觉模型编码，再经过同类 MLP 投影器。投影后的状态保留编码器的序列轴，并替换 Thinker 输入序列中连续的 `<|audio_pad|>` 或 `<|image_pad|>` embedding 位置。

Thinker 是完整的 MiniMind Transformer。Talker 是附加模块，包括 `num_talker_hidden_layers=4` 个 MiniMind block、自有 RMSNorm、Mimi-code embedding、codec projection 和音频 code 头。当加载不含 Talker 权重的 MiniMind 检查点且隐藏尺寸匹配时，Talker block 通过复制 Thinker 的最后四个 block 初始化。

**Talker的前向传播中，输入是两条投影流之和：`embed_proj(bridge_states)` 乘以可学习的文本 scale，加上 `codec_proj(talker_emb)` 乘以可学习的音频 scale。因此，Talker 同时读取语义状态和自回归 Mimi-code 历史，而不是简单地作为语言模型的后缀。**

**图 4：当前实现采用的训练流水线。** 训练脚本 `train_sft_omni.py` 在 T2A、I2T 和 A2A 数据上运行；`all` 模式更新完整模型，`vision_proj` 模式只进行视觉投影器对齐。SenseVoice 和 SigLIP2 在训练期间保持冻结。

音频 code 的输入和输出接口有意设计为非满秩。`TalkerEmbedding` 使用一个共享 embedding 表加每个 codebook 的低秩 adapter；`TalkerHead` 使用一个共享线性头加每个 codebook 的低秩 adapter。这样，模型仍能看到每个 codebook 的特定残差，同时不必把大型共享部分复制八次。

说话人控制被放在音频 code 缓冲区中，而不是文本流中。如果存在说话人嵌入，数据集会在参考 code 区域前保留一个位置，并在该位置的八个音频层全部填入 `<|audio_spk|>`；模型随后用投影后的 192 维 CAM++ 向量替换该位置的 Talker embedding。参考 Mimi code 在目标语音区域之前右对齐，并从音频损失中屏蔽。这样参考音频作为提示而非重建目标；当同一声音要复用到不同句子时，这一点很重要。

附录表 6 列出了每个模块、具体模型、关键配置和参数量。可训练参数量对绑定的 MiniMind token embedding 和文本 `lm_head` 去重；评测表保留实验层面的检查点统计口径，因此应将其理解为用于比较的模型规模标签，而不是表 6 的参数分解。

### 3.1 中间层桥接

小型全模态模型对桥接层非常敏感。embedding 层主要仍包含 token 身份和注入的多模态特征；它还没有积累足够上下文来处理发音、句法或跨模态指代。最后一层则有相反的偏置：它已经受到下一个文本 token 分类器的塑造，携带的是语言模型输出头的几何结构和 token 选择噪声，而不是 Talker 所需的声学条件。在一致性实验中，桥接层过深会增大 Talker CER，说明声学路径被已经过度专用于文本 logits 的状态条件化。

因此，MiniMind-O 从 Thinker 的中间层提取桥接状态，默认层为 `num_hidden_layers // 2 - 1`。这与 Qwen-Omni 风格 Thinker–Talker 系统中的中间层隐藏状态提取思路相近。在默认的八层 MiniMind 中，桥接状态在第 3 层之后提取。可学习的 `embed_proj` 将其映射到 Talker 隐藏空间，再与 codec 历史特征融合。表 2 的消融显示，较窄的 Talker 会先损失一致性，因此最终保留 768 维 Talker。

**图 5：MiniMind-O 的输入 token 布局。** 文本 token、音频占位符、图像占位符、说话人 token、参考 code 和目标音频 code 占据对齐的位置，使 Thinker 和 Talker 可以在统一的自回归计划下训练。

## 4 序列格式与流式解码

图 3 和图 5 展示了实际序列布局。每个训练样本是九路序列：八路音频 code 流加一路文本流。Thinker 读取文本流，其中重复的音频或图像占位符标记要被投影后的 SenseVoice 或 SigLIP2 状态替换的位置。Talker 读取八路音频流。在助手回答之前，音频流填充 pad，可选地放入右对齐参考 code，并可选地标记 speaker token 位置。回答开始后，这些位置承载目标 Mimi code。只有目标区域接收音频标签；参考区域和条件位置保持屏蔽。


对于文本 token `y₁:T` 和 Mimi code 矩阵 `a ∈ N^(8×T')`，MiniMind-O 优化联合的下一个 token 目标：

```math
L = L_text + λ_audio Σ(q=1..8) L_audio^(q)
```

其中 `q` 表示 Mimi codebook 层。无效位置和仅用于条件的位置会被屏蔽。数据集按 codebook 层错开音频目标：第 `q` 层从 `assistant_start + q + 1` 开始。在流式推理中，第一个生成的文本步没有音频输出，八个 codec 层按同样的延迟计划逐步可用。一旦完整的八层帧可用，Mimi code 就能增量解码成 24 kHz 波形，因此在完整文本回答结束前就可以开始播放。

这种格式在一个特定意义上比 ASR–LLM–TTS 级联系统更严格：Talker 对照的是 Thinker 自己的文本，而不是外部转写或人工参考。当数字、罕见名称或长从句没有正确读出时，错配可以追溯到共享的全模态路径。大型独立 TTS 模块可能会吸收一部分困难，而在这里行为会直接暴露。

## 5 训练流程

当前训练入口是 `train_sft_omni.py`。模式开关很小：`all` 同时更新可训练的 MiniMind、Talker 和投影器参数；`audio_proj` 冻结其余模型，只训练音频投影器；`vision_proj` 同理，只训练视觉投影器。当前 `train.sh` 对稠密版和 MoE 版都先在 `sft_t2a`、`sft_i2t` 和 `sft_a2a` 上进行全模型训练，然后进行一次仅投影器的 `sft_i2t` 训练。这与旧 README 中将 `t2t`、`t2a`、`a2a` 写成独立模式的描述不同；旧名称描述的是数据类型，而不是当前命令行模式接口。

所有本文结果都在一台工作站上产生，使用 4 张 NVIDIA RTX 3090（每张 24 GB），通过 `torchrun --nproc_per_node 4` 启动 PyTorch DDP。训练使用 bf16 混合精度、AdamW 优化器、每 GPU batch size 32、不进行梯度累积，并将梯度裁剪到 1.0。

各阶段配置如下：

| 阶段                    |    学习率 | 训练轮数/上下文              |
| ----------------------- | --------: | ---------------------------- |
| 全模型 T2A（`sft_t2a`） | `5×10^-6` | 1 个 epoch                   |
| 音频投影器 A2A          | `5×10^-4` | 1 个 epoch                   |
| 视觉投影器 I2T          | `5×10^-5` | 1 个 epoch                   |
| 全模型 A2A（`sft_a2a`） | `5×10^-5` | 3 个 epoch                   |
| 全模型 I2T              | `5×10^-6` | 1 个 epoch，768 token 上下文 |

每阶段大约耗时：T2A 45 分钟，音频投影器 A2A 25 分钟，三轮 A2A 75 分钟，每个 I2T 阶段 45 分钟。因此，在该硬件上，稠密版或 MoE 版完整训练周期均可在 4 小时内完成。正是约 0.1B 的激活参数规模使消费级 GPU 能够执行这一流程；在前沿规模下，同样的闭环无法用这么小的计算预算复现。

### 表 1：MiniMind-O 使用的主要训练数据集

音频时长由发布数据中预提取的 Mimi-code 统计量计算。

| 数据集    |    样本数 |   输入语音 |   输出语音 | 语音总时长 |
| --------- | --------: | ---------: | ---------: | ---------: |
| `sft_i2t` |   约 100K |          — |          — |          — |
| `sft_t2a` | 1,248,923 |          — | 1,636.01 h | 1,636.01 h |
| `sft_a2a` |   414,024 | 1,711.97 h |   423.40 h | 2,135.37 h |

公开数据集本身也是贡献的一部分，因为它固定了模型使用的精确序列和 codec 布局，而不是把复现工作留给私有预处理流程。`sft_t2a` 包含 1,248,923 个样本和 1,636.01 小时输出语音；`sft_a2a` 包含 414,024 个样本、1,711.97 小时输入语音和 423.40 小时输出语音。T2A 划分的中文和英文输出接近均衡：中文 45.7%、英文 46.5%、混合内容 7.8%。A2A 划分偏中文：中文 70.8%、英文 21.2%、混合内容 8.0%。这一分布会体现在行为中：短中文和英文回答通常稳定，较长英文语音更容易出现发音漂移和遗漏。



**图 6：`minimind-3o` 和 `minimind-3o-moe` 的文本到音频训练曲线。** 绘制曲线使用清理后的日志片段，移除了加载不兼容检查点导致的错误恢复区间。

**图 7：`minimind-3o` 和 `minimind-3o-moe` 的音频到音频训练曲线。** A2A 阶段在文本到音频学习之后训练，展示完整的语音输入到语音输出链路。

图 6 和图 7 展示两个语音生成阶段。T2A 曲线使用清理后的日志片段；一次从不兼容检查点恢复训练造成了损失尖峰，该区间没有用于绘图。MoE 版本总参数更多，但激活参数规模与稠密模型大致相同，因此这些曲线更适合观察容量分配，而不能据此宣称相同计算量下的优越性。

### 5.1 Talker 低秩接口消融

图 8 将 Talker 侧低秩接口与其余模型隔离。实验冻结 Thinker，并在相同的 A2A 子集上改变 `TalkerEmbedding` 和 `TalkerHead` adapter 的秩。统一提高两者的秩会改善收敛、最终音频损失和 codebook 准确率，但当 adapter 达到数百万参数后，收益逐渐变小。解耦实验更能说明问题：在相同设置下，将 `TalkerHead` 的秩从 16 提高到 256，比提高 `TalkerEmbedding` 的秩带来更大的改进。这与两个接口的作用一致：embedding 侧主要读取最近的 Mimi-code 历史，而 head 侧必须在完整音频词表上区分八个 codebook 的分布。

**图 8：Talker 低秩接口的秩消融。** 上排统一扫描 `TalkerEmbedding` 与 `TalkerHead` 的秩；下排将两者解耦。实线或柱表示音频损失，虚线或叠加标记表示音频准确率。统一秩的取值为 `r=4/16/64/256/768`，对应约 0.2M/0.7M/2.9M/11.8M/35.4M adapter 参数。中等秩已经可以获得大部分参数效率收益，且输出头秩比嵌入秩更重要。

### 5.2 Talker 隐藏维度消融

**表 2：Talker 隐藏维度消融。** 768 维 Talker 同时用于两个变体，因为它在取得最佳平均 CER 的同时保持了简单的 Thinker–Talker 维度接口。

| 变体  | Talker 隐藏维度 |           参数量 | 平均 CER↓ | 短回答↓ |    中等/长回答↓ |
| ----- | --------------: | ---------------: | --------: | ------: | --------------: |
| Dense |             768 |          115.29M |    0.0897 |  0.1528 | 0.0874 / 0.0675 |
| Dense |             512 |           96.13M |    0.1745 |  0.2709 | 0.2455 / 0.0976 |
| Dense |             384 |           88.72M |    0.2767 |  0.3904 | 0.1865 / 0.4046 |
| MoE   |             768 | 317.05M-A115.33M |    0.0900 |  0.2075 | 0.0533 / 0.0271 |
| MoE   |             512 |  261.32M-A96.17M |    0.1265 |  0.0711 | 0.1490 / 0.1464 |
| MoE   |             384 |  240.04M-A88.75M |    0.3280 |  0.3757 | 0.2777 / 0.4313 |

768 维 Talker 同时适用于两个变体，因为它在保持 Thinker–Talker 维度接口简单的同时给出最佳平均 CER。缩小到 512 或 384 虽然节省参数，却缩窄了每个 codebook 头看到的声学状态。由于 Mimi 预测本身是八层问题，这一瓶颈会在多个 codebook 间被放大。消融结果排除了一个简单的缩放假设：语义规划来自 Thinker，并不意味着 Talker 可以做得很薄。

## 6 评测

评测围绕演示中容易忽略的一致性属性构建。对每个提示，模型生成 Thinker 文本和 Talker 音频；音频由 Qwen3-ASR-Flash 转写，再与 Thinker 文本比较。内部一致性实验报告 CER；跨模型英文实验和视觉语言比较还报告词错误率（WER）。这些指标把自然度和偏好留给其他评测；本文只问一个更窄的问题：Talker 将隐藏状态转换为波形后，口头或书面输出是否仍与预期文本匹配？因此该协议依赖 ASR，不能解读为 MOS 或人类偏好研究。特别是，当波形正确但 ASR 将数字转写成英文单词时，数字格式会人为增大编辑距离。

### 6.1 声音克隆

**表 3：使用 CAM++ 说话人嵌入测量的声音克隆相似度。** “Previous baseline” 是开发阶段报告的仅使用参考 code 的旧设置。

| 模型              | 已见说话人↑ | 未见说话人↑ |  总体↑ |
| ----------------- | ----------: | ----------: | -----: |
| Previous baseline |      0.6150 |      0.5310 |      — |
| `minimind-3o`     |      0.6472 |      0.5654 | 0.5995 |
| `minimind-3o-moe` |      0.6267 |      0.5702 | 0.5937 |

已见划分使用发布在 `voices.pt` 中的五个内置声音：`dylan`、`eric`、`serena`、`uncle_fu`、`vivian`。未见划分使用 `voices_unseen.pt` 中的七个提示：`arthur`、`chelsie`、`cherry`、`ethan`、`jennifer`、`momo`、`moon`。每个声音使用相同的文本问题，只改变上下文说话人条件，即参考 Mimi code 和 192 维 CAM++ 向量。稠密版在已见说话人上略好，MoE 在未见说话人上略好，但总体差距很小。相较旧的仅参考 code 基线，稠密版已见声音从 0.6150 提高到 0.6472，MoE 未见声音从 0.5310 提高到 0.5702。

### 6.2 英文 T2A 跨模型一致性

**表 4：相同简短回答约束下的跨模型英文 T2A 一致性。** `minimind-3o` 的参数规模小于 Mini-Omni 和 Mini-Omni2，但差距主要集中在中等长度回答。

| 模型          | 参数量 | 平均 CER↓ | 平均 WER↓ |
| ------------- | -----: | --------: | --------: |
| Mini-Omni     |   0.5B |    0.0101 |    0.0185 |
| Mini-Omni2    |   0.5B |    0.0371 |    0.0431 |
| `minimind-3o` |   0.1B |    0.0964 |    0.0973 |

在相同简短回答协议下，`minimind-3o` 的平均 CER/WER 高于两个 Mini-Omni 基线，差距主要集中在 16–30 词的中等长度回答；按长度分桶的详细结果见附录表 8。

### 6.3 视觉语言能力

**表 5：使用 Qwen-VL-Plus 生成长度匹配参考答案的视觉语言比较。** CER/WER 偏高，是因为开放式图像描述允许很多有效的改写。

| 模型          | 参数量 | 平均 CER↓ | 平均 WER↓ |
| ------------- | -----: | --------: | --------: |
| Mini-Omni2    |   0.5B |    0.7609 |    0.9756 |
| `minimind-3o` |   0.1B |    0.8241 |    1.0293 |

Mini-Omni 不支持该路径，因此比较只包含 Mini-Omni2 和 `minimind-3o`。评测使用九张合成图像；对每个输出，Qwen-VL-Plus 为同一图像生成一个独立的长度匹配参考答案。由于开放式图像描述允许多种释义，两种正确描述可能几乎没有相同的 n-gram，所以绝对值偏高。在相同协议下，`minimind-3o` 落后于 Mini-Omni2，但仍处在同一数量级，而参数量约为后者的五分之一。

## 7 讨论与局限

MiniMind-O 的主要启示是，全模态闭环存在一个有意义的小模型工作区间：约 0.1B 激活参数就可以公开、检查完整的文本-语音-图像链路；训练数据可以用保留实际多模态布局的格式发布；八码本 embedding/head 接口不必在八个 codebook 之间完全复制；中间层桥接比最终下一个 token 预测状态为 Talker 提供了更干净的语义条件。即使模型距离前沿系统仍很远，这些仍是积极结果。

局限也很清楚。语音自然度和长文本稳定性落后于更大的语音-文本模型，中等长度英文回答是最明显的弱点。视觉路径使用冻结的 SigLIP2、64 个占位符和普通 MLP 投影器，更接近紧凑的视觉到语音路径，而不是大型 VLM 的替代品。声音克隆比早期的仅参考 code 基线有所提高，但仍高度依赖参考质量，以及生成音频是否足够干净以供说话人编码器读取。

MoE 版本更适合被视为容量分配实验，而不是最终的专家布局。评测也有意保持狭窄：主要自动分数衡量转写一致性，而不是人类自然度、负载下的延迟、安全行为或对噪声远场语音的鲁棒性。本文的论断同样是有边界的：MiniMind-O 没有被宣称为前沿系统的竞争者；它的价值在于，不把关键选择隐藏在规模之后，就能复现和检查完整全模态闭环。

## 8 结论

本文介绍了 MiniMind-O：一个约 0.1B 参数规模、接收文本/语音/图像输入并输出流式语音的开放全模态模型。当前代码将完整 MiniMind Thinker、独立四层 Talker、中间层语义桥接、基于 MLP 的音频/视觉投影、Mimi-code 语音生成，以及基于公开 T2A、I2T、A2A 数据的分阶段 SFT 结合起来。稠密版和 MoE 版在简短回答设置下都保持了可用的 Thinker–Talker 一致性，支持说话人条件生成，并能运行基本的视觉语言到语音交互。

更广泛的结论是，小型全模态模型可以成为受控的研究对象：通过公开数据、中间隐藏桥接和低秩的 codebook 特定 embedding/head adapter，可以将完整闭环做到足够参数高效，直接进行研究。因此，MiniMind-O 的贡献是一个用于分析原生语音全模态设计的可复现小规模基线，而不仅仅是一个能运行的演示。相同实现暴露出的剩余缺口，也使小模型规模适合分析，而不只是提高部署效率。

## 附录 A：模块与评测细节

本附录收集正文引用的详细表格。表 6 枚举当前 MiniMind-O 实现中的每个模块、具体模型、关键超参数和参数量。可训练参数量对共享的 MiniMind token embedding 与文本 `lm_head` 去重；冻结模块按原样加载，训练期间不更新。

### 表 6：当前实现使用的主要模块

可训练组件的参数量取自当前 PyTorch 模块；外部感知模型和 codec 模型被冻结，不计入 MiniMind-O 的激活参数量。

| 模块       | 具体模型/层              | 关键配置                                                                         | 状态/参数量（Dense / MoE） |
| ---------- | ------------------------ | -------------------------------------------------------------------------------- | -------------------------: |
| Thinker    | MiniMind Transformer     | 8 层，隐藏 768，8 个 query head，4 个 KV head，词表 6400                         |   可训练，63.91M / 198.42M |
| Talker     | 独立 MiniMind block      | 4 层，隐藏 768，音频词表 2112，8 个 codebook 头，rank-256 embedding/head adapter |   可训练，47.05M / 114.30M |
| 音频投影器 | `MMAudioProjector`       | `LayerNorm(512) – Linear – GELU – Linear` 到隐藏 768                             |              可训练，0.99M |
| 视觉投影器 | `MMVisionProjector`      | `LayerNorm(768) – Linear – GELU – Linear` 到隐藏 768                             |              可训练，1.18M |
| 音频编码器 | SenseVoice-Small         | 50 个 encoder block，输出 512，16 kHz 前端                                       |              冻结，234.00M |
| 视觉编码器 | SigLIP2 base patch32-256 | 12 层，隐藏 768，12 个 head，64 个图像 token                                     |               冻结，94.55M |
| 语音 codec | Mimi                     | 8 个 codebook，大小 2048，12.5 Hz 帧率，24 kHz 波形                              |               冻结，96.15M |
| 说话人条件 | CAM++ embedding          | 192 维向量，经 `spk_proj` 投影                                                   |   预计算，不在线运行编码器 |

### 表 7：逐说话人声音克隆相似度

| 划分 | 说话人   | `minimind-3o`↑ | `minimind-3o-moe`↑ |
| ---- | -------- | -------------: | -----------------: |
| 已见 | dylan    |         0.6997 |             0.6837 |
| 已见 | eric     |         0.5289 |             0.4232 |
| 已见 | serena   |         0.7092 |             0.7041 |
| 已见 | uncle_fu |         0.7241 |             0.7337 |
| 已见 | vivian   |         0.5744 |             0.5888 |
| 未见 | arthur   |         0.7171 |             0.6750 |
| 未见 | chelsie  |         0.6437 |             0.6240 |
| 未见 | cherry   |         0.5689 |             0.5678 |
| 未见 | ethan    |         0.4783 |             0.4847 |
| 未见 | jennifer |         0.4749 |             0.4003 |
| 未见 | momo     |         0.6470 |             0.5720 |
| 未见 | moon     |         0.4282 |             0.6673 |

五个已见声音来自 `voices.pt`，七个未见声音来自 `voices_unseen.pt`，训练期间从未见过。每个声音使用同一组文本问题，只改变上下文说话人条件。最佳单个声音（`uncle_fu`、`serena`、`arthur`）至少在一个变体中超过 0.70 余弦相似度；最低离群点（MoE 的 `eric`、稠密版的 `moon`）通常与说话人编码器处理前已经退化的生成音频质量同时出现。

### 表 8：按长度分桶的英文 T2A 比较

每项为 CER / WER，括号内为样本数。

| 长度           |               Mini-Omni |              Mini-Omni2 |           `minimind-3o` |
| -------------- | ----------------------: | ----------------------: | ----------------------: |
| 短（≤15 词）   |  0.0195 / 0.0384（n=8） | 0.0503 / 0.0584（n=14） |  0.0531 / 0.0417（n=8） |
| 中（16–30 词） | 0.0038 / 0.0052（n=12） |  0.0062 / 0.0076（n=6） | 0.1327 / 0.1420（n=11） |
| 长（31–60 词） |                       — |                       — |  0.0431 / 0.0508（n=1） |

在相同协议下，`minimind-3o` 在短回答（≤15 词）上与 Mini-Omni2 具有竞争力，但在中等长度回答（16–30 词）上落后；这时 Talker 必须在整个从句中持续保持发音和词汇一致性。

### 表 9：逐问题英文 T2A 比较

所有问题都带有前缀 “Answer briefly in one short sentence”（请用一个简短句子回答）。每个单元格为 CER / WER；CER > 0.3 的条目通常由数字拼写或专有名词变体等表面转写差异造成。

| 编号与问题              |         Mini-Omni |        Mini-Omni2 |     `minimind-3o` |
| ----------------------- | ----------------: | ----------------: | ----------------: |
| 00 今天过得怎么样？     |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 01 能讲个笑话吗？       |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 02 法国首都是什么？     |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 03 如何冲一杯咖啡？     |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 04 光速是多少？         |     0.000 / 0.000 |     0.382 / 0.286 |     1.410 / 1.471 |
| 05 解释什么是 AI。      |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 06 世界最高的山？       |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 07 太阳系有多少颗行星？ |     0.156 / 0.125 |     0.303 / 0.250 |     0.000 / 0.000 |
| 08 彩虹是怎么形成的？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 09 推荐一本好书？       |     0.000 / 0.182 |     0.000 / 0.182 |     0.000 / 0.000 |
| 10 地球上最大的海洋？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 11 光合作用如何进行？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.017 / 0.045 |
| 12 规律运动有什么好处？ |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 13 谁发明了电话？       |     0.000 / 0.000 |     0.000 / 0.000 |     0.425 / 0.333 |
| 14 生命的意义是什么？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 15 飞机如何保持飞行？   |     0.000 / 0.000 |     0.019 / 0.100 |     0.033 / 0.045 |
| 16 病毒和细菌的区别？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 17 解释区块链技术。     |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| 18 改善睡眠的三个建议？ |     0.045 / 0.063 |     0.037 / 0.045 |     0.043 / 0.051 |
| 19 天空为什么是蓝色？   |     0.000 / 0.000 |     0.000 / 0.000 |     0.000 / 0.000 |
| **平均**                | **0.010 / 0.019** | **0.037 / 0.043** | **0.096 / 0.097** |

20 个问题中有 14 个对三个模型都得到 CER=0。少数高 CER 离群点主要来自表面形式错配，而非明确的发音失败。例如问题 04 涉及数字“299,792,458”，ASR 可能把语音转成 “two hundred ninety-nine million …”，从而增大字符距离；问题 13 的专有名词也有同样的指标敏感性。

### 表 10：逐样本视觉语言比较

每个单元格为“输出长度 / 参考长度 / CER / WER”。

| 编号 | Mini-Omni2                  | `minimind-3o`               |
| ---- | --------------------------- | --------------------------- |
| 00   | 158 / 128 / 0.7976 / 1.0391 | 111 / 103 / 0.7883 / 0.9709 |
| 01   | 260 / 233 / 0.7273 / 0.9914 | 109 / 87 / 0.8013 / 1.0920  |
| 02   | 110 / 89 / 0.7957 / 0.9888  | 126 / 98 / 0.8728 / 1.0816  |
| 03   | 79 / 84 / 0.7408 / 0.9286   | 113 / 104 / 0.8031 / 0.9519 |
| 04   | 94 / 102 / 0.7273 / 0.8529  | 130 / 101 / 0.8531 / 1.0594 |
| 05   | 91 / 72 / 0.7209 / 1.0556   | 115 / 91 / 0.8629 / 1.0989  |
| 06   | 187 / 194 / 0.7293 / 0.9021 | 119 / 118 / 0.7551 / 0.8814 |
| 07   | 295 / 267 / 0.7682 / 0.9625 | 114 / 89 / 0.8548 / 1.0449  |
| 08   | 143 / 118 / 0.8414 / 1.0593 | 137 / 109 / 0.8254 / 1.0826 |

## 附录 B：定性示例

本附录展示 MiniMind-O 支持的三种交互模式：带 barge-in 打断的实时流式交互（图 9）、音频到音频对话（图 10）和图像条件语音生成（图 11）。示例由 `minimind-3o` 生成；发布包中的 HTML demo 页面提供了所展示案例的可播放音频。

**图 9：实时交互界面。** 流式语音生成允许在解码继续时播放；由 VAD 触发的 barge-in 可以在检测到新的用户回合时停止当前输出。用户说完后，Thinker 先进行语义侧 prefill，Talker 开始生成音频 code，Mimi 解码器随着新 code 帧到达写出 24 kHz 波形。当模型播放期间用户再次说话，系统检测到新的语音事件，放弃当前生成，开始新的 prefill–reply 循环。这里并不是声称具有人类级的全双工轮流对话能力；打断检测仍然只是简单的 VAD 阈值，而不是对重叠语音的语义理解。它是一个规模更小但实用的工程闭环：系统可以离开说话状态，接受新请求，并且无需等待上一段波形播放完就产生下一条回答。

**图 10：定性 A2A 示例。** 模型接收真实语音，同时返回对齐的文本和语音输出，展示完整的语音输入到语音输出链路。短的助手式对话最稳定：Thinker 生成紧凑的语义回答，Talker 可以在音频 code 错误累积前完成渲染。中文解释型提示通常保持连贯，英文回答在发音和节奏上变化更大；较长回答仍然可生成，但会暴露与表 4 相同的发音漂移和少量词语遗漏问题。

**图 11：图像到音频定性示例。** 图像特征投影到 Thinker，生成的回答再通过 Talker 渲染为语音。该路径把视觉编码、文本生成和语音渲染连接到一个流水线中。示例说明系统可以让语音受图像内容条件化，同时也暴露小模型的典型错误：有些输出能抓住场景的大致内容，有些会替换主要对象或混淆属性，例如动物类别或车辆类型。这些错误与 64 个图像占位符预算和 0.1B 基础规模一致，因此示例应被理解为小型全模态流水线可以端到端运行的证据，而不是开放式图像描述能力的上限。

## 参考文献

1. Keyu An 等。*FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction between Humans and LLMs*。arXiv:2407.04051，2024。
2. Jinze Bai 等。*Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities*。arXiv:2308.12966，2023。
3. Jade Copet 等。*Simple and Controllable Music Generation*。NeurIPS 36，2024。
4. Alexandre Défossez 等。*High Fidelity Neural Audio Compression*。arXiv:2210.13438，2022。
5. Alexandre Défossez 等。*Moshi: A Speech-Text Foundation Model for Real-Time Dialogue*。arXiv:2410.00037，2024。
6. Qingkai Fang 等。*LLaMA-Omni: Seamless Speech Interaction with Large Language Models*。arXiv:2409.06666，2024。
7. Chaoyou Fu 等。*VITA: Towards Open-Source Interactive Omni Multimodal LLM*。arXiv:2408.05211，2024。
8. Jingyao Gong。*MiniMind: Train a Small Language Model from Scratch*。GitHub，2024。
9. Jingyao Gong。*MiniMind-V: Train a Small Vision-Language Model from Scratch*。GitHub，2025。
10. Jingyao Gong。*MiniMind-O: Train a Tiny Omni Model from Scratch*。GitHub，2026。
11. Yitian Gong 等。*MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models*。arXiv:2602.10934，2026。
12. Ailin Huang 等。*Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction*。arXiv:2502.11946，2025。
13. Junnan Li 等。*BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*。ICML，2023。
14. Tianpeng Li 等。*Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction*。arXiv:2502.17239，2025。
15. Haotian Liu 等。*Improved Baselines with Visual Instruction Tuning*。CVPR，2024。
16. Tu Anh Nguyen 等。*Spirit-LM: Interleaved Spoken and Written Language Model*。TACL，13:30–52，2025。
17. OpenAI。*Hello GPT-4o*，2024。
18. Alec Radford 等。*Learning Transferable Visual Models from Natural Language Supervision*。ICML，2021。
19. Hubert Siuzdak。SNAC 项目，2024。
20. Michael Tschannen 等。*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*。arXiv:2502.14786，2025。
21. Chengyi Wang 等。*Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*。arXiv:2301.02111，2023。
22. Hui Wang 等。*CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking*。arXiv:2303.00332，2023。
23. Peng Wang 等。*Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution*。arXiv:2409.12191，2024。
24. Zhifei Xie、Changqiao Wu。*Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming*。arXiv:2408.16725，2024。
25. Zhifei Xie、Changqiao Wu。*Mini-Omni2: Towards Open-Source GPT-4o with Vision, Speech and Duplex Capabilities*。arXiv:2410.11190，2024。
26. Jin Xu 等。*Qwen2.5-Omni Technical Report*。arXiv:2503.20215，2025。
27. Jin Xu 等。*Qwen3-Omni Technical Report*。arXiv:2509.17765，2025。
28. Aohan Zeng 等。*GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot*。arXiv:2412.02612，2024。
