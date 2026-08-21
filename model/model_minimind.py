# ============================================================================
# 导入基础库和 PyTorch 相关模块
# ============================================================================
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN          # 激活函数映射表
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast  # MoE 输出结构

# ============================================================================
# 配置类：MiniMindConfig
# 继承自 HuggingFace 的 PretrainedConfig，方便与 transformers 库集成
# ============================================================================
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"   # 模型类型标识

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 基础配置
        self.hidden_size = hidden_size                # 隐藏层维度
        self.num_hidden_layers = num_hidden_layers    # Transformer 层数
        self.use_moe = use_moe                        # 是否使用混合专家（MoE）

        # 从 kwargs 中获取可选参数，若未提供则使用默认值
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)   # 是否使用 Flash Attention

        # 注意力头配置
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # GQA 分组数
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # 激活函数与 FFN 维度（用 pi 做缩放使其更规整）
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # 位置编码与归一化
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # 权重绑定（输入输出 embedding 共享）
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # YaRN 动态缩放（用于扩展上下文长度）
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # MoE 专用配置（仅当 use_moe=True 时生效）
        self.num_experts = kwargs.get("num_experts", 4)                     # 专家总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)     # 每个 token 选几个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)            # 是否归一化专家权重
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 辅助损失系数


# ============================================================================
# RMSNorm：Root Mean Square Layer Normalization
# 比 LayerNorm 更轻量，无均值中心化，仅缩放
# ============================================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习的缩放参数

    def norm(self, x):
        # RMS = sqrt(mean(x^2) + eps)，然后用 x / RMS
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转换为 float 计算，再转回原 dtype 以保持数值稳定性
        return (self.weight * self.norm(x.float())).type_as(x)


# ============================================================================
# 旋转位置编码（RoPE）预计算函数
# 支持 YaRN 缩放，用于扩展上下文窗口
# ============================================================================
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 sin/cos 表，供后续 apply_rotary_pos_emb 使用
    dim: 每个头的维度（head_dim）=96
    end: 最大序列长度
    rope_base: RoPE 基值
    rope_scaling: YaRN 缩放配置（可选）
    """
    # 基础频率：1 / (base^(2i/dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    # 若启用 YaRN 缩放，则根据配置重新调整频率
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 仅当当前最大长度超过原始长度时才应用缩放
        if end / orig_max > 1.0:
            # 计算维度边界：哪些维度需要快速/慢速缩放
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 线性斜坡 ramp: 在 low~high 之间从 0 到 1
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 频率调整：freqs = freqs * (1 - ramp + ramp / factor)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置索引 t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 外积：freqs_cis = t * freqs （形状：[end, dim//2]）
    freqs = torch.outer(t, freqs).float()

    # 分别拼接正弦、余弦，并乘以注意力因子（通常为1）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


# ============================================================================
# 应用旋转位置编码到 q 和 k 上
# ============================================================================
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, k: 形状 (batch, seq_len, num_heads, head_dim)
    cos, sin: 形状 (seq_len, head_dim) 或 (1, seq_len, head_dim)
    """
    def rotate_half(x):
        # 将后半部分取负拼接到前半部分，实现复数乘法
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


# ============================================================================
# 重复 KV 头以实现 GQA（分组查询注意力）
# ============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 x 中的每个头复制 n_rep 次，用于匹配 Q 的头数"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 扩展并重塑
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


# ============================================================================
# 多头注意力（支持 GQA、Flash Attention、KV cache）
# ============================================================================
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 头数配置
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads          # Q 的头数
        self.n_local_kv_heads = self.num_key_value_heads         # KV 的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个 KV 头要重复的次数
        self.head_dim = config.head_dim
        self.is_causal = True

        # 投影层（无偏置）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Q/K 的 RMSNorm（改善训练稳定性）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 是否启用 Flash Attention（PyTorch 2.0+ 内置）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # 线性投影
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Q/K 归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 应用旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 拼接过去的 KV（用于自回归生成时的缓存）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 调整维度：将头维度放到第2维，方便计算
        xq = xq.transpose(1, 2)                     # (bs, n_local_heads, seq_len, head_dim)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 使用 Flash Attention 或标准实现
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention（训练或推理时均可用）
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )
        else:
            # 标准点积注意力
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码（仅对当前生成的这部分加上三角掩码）
            if self.is_causal:
                # 用 -inf 遮住未来位置
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 外部注意力掩码（如 padding）
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 恢复形状并输出
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ============================================================================
# 标准前馈网络（SwiGLU 风格）
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # SwiGLU 结构： gate_proj * up_proj，再 down_proj
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]   # 例如 SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 混合专家（MoE）前馈网络
# 包含 router 和多个专家，支持负载均衡辅助损失
# ============================================================================
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由门控：输出每个专家的得分
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 专家列表：每个专家是一个标准 FeedForward
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # 展平为 (batch*seq_len, hidden_dim)

        # 路由分数（softmax 归一化）
        scores = F.softmax(self.gate(x_flat), dim=-1)   # (N, num_experts)
        # 选择 top-k 个专家及其权重
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 可选：将 topk 权重归一化（使得和为1）
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 初始化输出 tensor
        y = torch.zeros_like(x_flat)

        # 逐个专家处理，并将结果加权累加到对应 token 位置
        for i, expert in enumerate(self.experts):
            # 哪些 token 选中了当前专家
            mask = (topk_idx == i)   # shape (N, k)
            if mask.any():
                # 找到至少有一个 token 选中该专家的行索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 提取这些 token 对应的权重（每个 token 选到该专家时可能有权重）
                weight = topk_weight[mask].view(-1, 1)
                # 将专家输出乘以权重后累加到 y 中
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时，若有专家从未被选中，为防止梯度断开，加一个微小贡献
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 计算辅助损失（负载均衡损失），训练时使用
        if self.training and self.config.router_aux_loss_coef > 0:
            # 每个专家被选中的频率（token 级别）
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)   # (num_experts,)
            # aux_loss = (load * scores.mean(0)).sum() * num_experts * coeff
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# Transformer 层（包含自注意力 + 前馈，支持 MoE）
# ============================================================================
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置选择标准 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 残差连接：先 norm + self-attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # 再 norm + FFN（MoE 或标准）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ============================================================================
# MiniMind 主模型（不含输出头）
# ============================================================================
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 层堆叠
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 的 sin/cos 表，并注册为 buffer（不参与训练，也不保存到 state_dict）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        # 兼容 transformers 5.x 传递的 past_key_values 格式
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算起始位置（用于 cache）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 嵌入层 + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 若 RoPE 表在 meta device 初始化时丢失（transformers>=5.x），重新计算
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        # 取出当前序列对应的 RoPE 片段
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层 forward，收集每层的 KV cache
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终 LayerNorm
        hidden_states = self.norm(hidden_states)

        # 收集所有 MoE 层的辅助损失（如果有）
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


# ============================================================================
# 因果语言模型（用于自回归生成）
# 继承 PreTrainedModel 和 GenerationMixin 以兼容 HuggingFace 接口
# ============================================================================
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}   # 权重绑定

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 若启用权重绑定，则 lm_head 与 embedding 共享参数
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.post_init()   # 执行一些初始化（如权重初始化、梯度检查点等）

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 调用模型主体
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # 根据 logits_to_keep 选择输出的 logits 切片（用于加速）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 计算交叉熵损失（shift 后预测下一个 token）
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # 使用 MoE 的输出类（兼容 transformers）
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )

    # ========================================================================
    # 自定义 generate 方法（覆盖父类，实现温度、top-p、top-k、重复惩罚等）
    # 参考：https://github.com/jingyaogong/minimind/discussions/611
    # ========================================================================
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 输入处理
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            # 仅在增量部分进行前向（使用 cache）
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs
            )

            # 更新 attention_mask（追加1）
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)

            logits = outputs.logits[:, -1, :] / temperature   # 温度缩放

            # 重复惩罚（repetition penalty）
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)

            # Top-k 过滤
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_k_vals[..., -1, None]] = -float('inf')

            # Top-p（nucleus）过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样或贪心
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 若已结束则强制生成 EOS
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            # 更新序列和 cache
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            # 检查是否全部结束
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()

        # 若需要返回 KV cache
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids