# 调试代码笔记
[text](../model/model_omni.py)357行
先判空，防止use_cache=False时报错
if present is not None:
    k, v = present
    print("key shape:", k.shape)
    print("value shape:", v.shape)
    print("batch:", k.shape[0])
    print("num_heads:", k.shape[1])
    print("cached_seq_len(已经缓存token数量):", k.shape[2])
    print("head_dim:", k.shape[3])
else:
    print("present is None，use_cache关闭")


看presents
print("\n====全部层收集完毕 presents====")
print(f"总层数:{len(presents)}")
for idx, pr in enumerate(presents):
    if pr is not None:
        k_pr, v_pr = pr
        print(f"layer {idx}: k {k_pr.shape} v {v_pr.shape}")
    else:
        print(f"layer {idx}: present = None (use_cache关闭)")

看talker_pos_emb
[text](../model/model_omni.py)376行
cos_emb, sin_emb = talker_pos_emb
print(f"cos_emb.shape = {cos_emb.shape}")
print(f"sin_emb.shape = {sin_emb.shape}")
print(f"start_pos={start_pos}, seq_length={seq_length}")


[text](../model/model_omni.py)392行
遍历打印audio_logits的每一个shape
for idx, logit_tensor in enumerate(audio_logits):
    print(f"codebook {idx}: shape = {logit_tensor.shape}")