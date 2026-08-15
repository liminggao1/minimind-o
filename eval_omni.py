import argparse
import os
import random
import time
import warnings
import torch
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel
from model.model_omni import MiniMindOmni, OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed, log_model_params

warnings.filterwarnings("ignore")


def init_model(args):
    # 从model目录加载
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 2. 判断加载来源：如果是本地训练好的模型（路径中包含 'model' 字样）
    if "model" in args.load_from:
        moe_suffix = "_moe" if args.use_moe else ""
        # ./out/sft_omni_768_moe.pth
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        # 实例化自定义的多模态模型 MiniMindOmni
        model = MiniMindOmni(
            # 传入配置对象，包含隐藏层大小、层数、是否使用 MoE
            OmniConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
            ),
            # 指定音频编码器（SenseVoiceSmall）和视觉编码器（SigLIP）的本地路径
            audio_encoder_path="./model/SenseVoiceSmall",
            vision_model_path="./model/siglip2-base-p32-256-ve",
        )

        # 加载本地保存的模型权重（checkpoint）
        # strict=False 表示允许部分键不匹配（例如用于微调时加载部分权重）
        model.load_state_dict(
            torch.load(
                ckp, map_location=args.device
            ),  # 加载 .pth 文件，并映射到指定设备
            strict=False,
        )

    else:
        # 3. 否则，从 Hugging Face 加载预训练模型（信任远程代码）
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            trust_remote_code=True,  # 允许执行自定义建模代码（常用于非官方模型）
        )
        # 手动为模型挂载音频编码器和处理器（因为预训练模型可能不包含多模态部分）
        model.audio_encoder, model.audio_processor = MiniMindOmni.load_sensevoice("./model/SenseVoiceSmall")
        # 手动挂载视觉编码器和处理器
        model.vision_encoder, model.vision_processor = MiniMindOmni.load_vision("./model/siglip2-base-p32-256-ve")

    # 4. 打印/记录模型参数量（用于调试或监控）
    log_model_params(model)

    # 5. 将音频编码器（如果存在）移动到指定设备（GPU/CPU）
    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)

    # 6. 将视觉编码器（如果存在）移动到指定设备
    if model.vision_encoder is not None:
        model.vision_encoder.to(args.device)

    # 7. 加载 Mimi 音频编解码模型（用于音频 token 化），并设置为评估模式
    model.mimi_model = MimiModel.from_pretrained("./model/mimi").eval()

    # 8. 将整个模型转为半精度（half），设置为评估模式，并移动到目标设备
    #    最后返回模型和分词器
    return model.half().eval().to(args.device), tokenizer


def eval_sample(
    model,
    tokenizer,
    args,
    idx,
    prompt,
    audio_inputs,#输入的音频特征（如音频token或编码器输出）
    output_name,#输出音频文件名（如 .mp3）
    pixel_values=None,#图像输入（可选）
    history=None,#对话历史（消息列表，可选）
    audio_lens=None,#音频帧长度（用于padding等）
    ref_codes=None,#参考音频的Mimi codes（用于语音克隆）
    spk_emb=None,#说话人嵌入（用于语音克隆）
):
    # 1. 构建对话消息：如果有历史记录则拼接，否则只使用当前用户消息
    messages = (history or []) + [{"role": "user", "content": prompt}]

    # 2. 应用聊天模板将消息转换为模型输入的文本字符串（添加生成提示，如<|im_start|>等）
    inputs_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 暂不tokenize，只返回字符串
        add_generation_prompt=True,  # 添加生成前缀（如"<|im_start|>assistant\n"）
        open_thinking=bool(args.open_thinking),  # 是否启用思维链（根据模型自定义）
    )

    # 3. 对文本进行tokenize，并转换为模型输入张量（形状 [1, seq_len]）
    #    例如：inputs_text = "你好，请介绍一下自己。" -> tokenizer得到 [101, 234, 567, 890,....] 的token ids.这是 Python 的 列表（List）。它的 长度（len） 是 15。此时，它只是一串普通数字，没有形状（Shape）的概念，也不能做矩阵运算.
    # 然后变成tensor([101, 234, 567, 890,....])的张量，
    # [None, ...]:变成了 torch.Size([1, 15]),这就是平常理解的 [1, 15]：1 代表批次（Batch），15 代表序列长度（Seq_len）
    x = torch.tensor(
        tokenizer(inputs_text).data["input_ids"], dtype=torch.long, device=args.device
    )[None, ...]  # 增加batch维度，形状变为 [1, seq_len]

    audio_frames = []  # 用于存储生成过程中产生的音频帧（Mimi codes）
    # 每个元素是一个长度为8的列表（或元组），代表一帧音频token

    # 4. 进入推理模式（不计算梯度）
    with torch.no_grad():
        # 调用模型的generate方法进行自回归生成
        # 返回一个生成器，逐步产出 (文本token列表, 音频帧)
        res_y = model.generate(
            x,  # 输入文本token ids，形状 [1, seq_len]
            tokenizer.eos_token_id,  # 结束标记id
            max_new_tokens=args.max_new_tokens,  # 最大生成token数，例如512
            temperature=args.temperature,  # 采样温度，值越大（如 1.0），输出越随机
            top_p=args.top_p,  # nucleus采样阈值，只从累积概率达到 top_p 的最小词汇集合里采样，提高生成质量
            stream=True,  # 流式输出True 表示“吐一个词就立刻返回一个词”，用于打字机效果；False 表示思考完所有词才一次性返回。
            return_audio_codes=True,  # 同时返回音频codes
            open_thinking=bool(args.open_thinking),  # 是否启用思维链
            audio_inputs=audio_inputs,  # 输入音频特征，可能形状 [batch, T, feat_dim] T 是时间帧数，F 是特征维度
            audio_lens=audio_lens,  # 输入音频长度（每帧有效长度）
            pixel_values=pixel_values,  # 输入图像张量，形状 [batch, C, H, W]（例如 [1, 3, 224, 224]）3 是 RGB 通道，224 是宽高。模型会通过视觉编码器看懂图片内容
            ref_codes=ref_codes,  # 参考音频的Mimi codes，形状 [1, 8, T_ref]8,是特征维度,T：比如 Mimi 的帧率是 80Hz（每秒 80 帧）。如果参考音频长 3 秒，那么 T_ref = 240。如果长 0.5 秒，T_ref = 40。
            spk_emb=spk_emb,  # 说话人嵌入，形状 [1, emb_dim]192、256 或 512。这个维度是固定的，不随音频时长变化，这是一个经过说话人验证模型（如 ECAPA-TDNN 或 WavLM）提取出来的高维浮点数特征向量。
        )

        # 5. 处理流式输出：打印文本，收集音频帧
        print("📒 [Thinker]: ", end="", flush=True)
        history_idx = 0  # 用于跟踪已打印的文本长度（去重）

        # y 是张量，内容是tensor([[101, 102, 103]])，形状为 (1, 3)
        # audio_frame 是 Python 列表，内容为 [1024, 2048, 512, 789, 111, 222, 333, 444],形状为 (1, 8)
        for y, audio_frame in res_y:
            if y is not None:
                # y 是当前步生成的token id列表，形状 [1, 3]，answer=“你好吗？”
                # audio_frame=none
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                # 如果非空且不是乱码字符，则增量打印新内容（从上次打印的索引开始）
                if answer and answer[-1] != "�":
                    # print("你好吗"[0:]) = 打印 "你好吗"
                    print(answer[history_idx:], end="", flush=True)
                    history_idx = len(answer)  # history_idx=3
            if audio_frame:
                # audio_frame 是一个长度为8的列表（Mimi codes的一帧）
                # 例如 [1024, 2048, 512, ...] 共8个整数
                audio_frames.append(audio_frame)
        print()  # 换行

        # 6. 处理音频输出
        if audio_frames:
            print(f"🎹 [Talker]: {len(audio_frames)} frames", end=" ")
            if args.decode_audio:
                # 如果启用音频解码，尝试将收集到的音频codes解码为wav并转换为mp3
                try:
                    # 过滤掉无效帧（确保每帧长度为8）
                    codes = [f for f in audio_frames if f and len(f) == 8]
                    if not codes:
                        print("⚠️  生成的Mimi codes为空，跳过保存。")
                        return
                    # 将codes转换为张量：假设 codes 是包含 N 个长度为8的列表，形状为 (N, 8)
                    # 转换后形状：(8, N) -> (1, 8, N)
                    mimi_codes = (
                        torch.tensor(codes, dtype=torch.long)
                        .T.unsqueeze(0)
                        .to(args.device)
                    )

                    # Mimi模型的vocab_size通常为2048（或2049），此处将超出范围的值置零防止解码时出现未知token导致错误
                    # 广播机制逐元素比较
                    filtered = torch.where(
                        mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes
                    )

                    # 调用Mimi模型解码生成音频波形
                    # decode() 输入形状 [1, 8, N] -> 输出包含 .audio_values，形状 [1, 1, T]，1段音频，1个声道（单声道），T个采样点（“血肉”）
                    audio = model.mimi_model.decode(filtered).audio_values

                    # 构造输出路径（如 output_dir/sample.mp3）
                    output_path = os.path.join(args.output_dir, output_name)
                    wav_path = output_path.rsplit(".", 1)[0] + ".wav"  # 临时wav文件

                    # 使用soundfile将音频保存为wav（采样率24000 Hz）
                    # audio.squeeze() 去除batch和channel维度，得到形状 [T] 的一维数组
                    sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)

                    # 用pydub将wav转为mp3（64k比特率）
                    AudioSegment.from_wav(wav_path).export(
                        output_path, format="mp3", bitrate="64k"
                    )

                    # 删除临时wav文件
                    os.remove(wav_path)

                    print(f"| Audio decoded to: {output_path}")
                except Exception as e:
                    print(f"⚠️  保存音频失败: {str(e)}")
            else:
                print("(decode_audio=off)\n")
        # 如果没有音频帧，函数正常结束（不输出音频）


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O Chat")
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument("--weight", default="sft_omni", type=str, help="权重名称前缀")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构"
    )
    parser.add_argument("--max_new_tokens", default=512, type=int, help="最大生成长度")
    parser.add_argument(
        "--temperature", default=0.7, type=float, help="Thinker生成温度"
    )
    parser.add_argument("--top_p", default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument(
        "--output_dir", default="./output_audio/", type=str, help="输出音频保存目录"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    parser.add_argument(
        "--audio_dir", default="./dataset/eval_omni/", type=str, help="测试音频目录"
    )
    parser.add_argument(
        "--image_dir", default="./dataset/eval_omni/", type=str, help="测试图像目录"
    )
    parser.add_argument(
        "--open_thinking",
        default=0,
        type=int,
        help="是否开启思考模式（0=否，1=是）（思考模式下禁用audio输出）",
    )
    parser.add_argument(
        "--decode_audio", default=1, type=int, help="是否解码音频输出（0=否，1=是）"
    )
    parser.add_argument(
        "--mode",
        default="0",
        type=str,
        help="评估模式：-1=all 0=text 1=multi 2=audio 3=clone 4=image 5=mix（逗号组合，如 2,5）",
    )
    parser.add_argument(
        "--prompt_lang",
        default=0,
        type=int,
        choices=[0, 1, 2],
        help="问题语言：0=英文 1=中文 2=英文+中文",
    )
    # 解析命令行传入的全部参数，得到args对象，后续所有配置都从args读取
    args = parser.parse_args()
    # 1.把逗号全部删掉；2.如果传入-1，则替换为字符串"012345"代表开启全部评估模式
    modes = set(args.mode.replace(",", "").replace("-1", "012345"))
    # 创建output_audio输出目录；exist_ok=True：目录已存在不会抛异常，避免重复创建报错
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, tokenizer = init_model(args)
    setup_seed(int(time.time()) % 31415926)

    if "0" in modes:
        print("\n\n==================== text -> {text, audio} ====================")
        test_prompts_en = [
            "Tell me an interesting fact about space.",
            "How do I make a cup of coffee?",
            "What's the weather like today?",
            "Will it rain tomorrow?",
            "Tell me a joke.",
            "Can you sing a song for me?",
            "Please introduce yourself.",
        ]
        test_prompts_zh = [
            "告诉我一个关于太空的有趣事实。",
            "如何制作一杯咖啡？",
            "今天的天气怎么样？",
            "明天会下雨吗？",
            "给我讲个笑话吧",
            "你能为我唱首歌吗？",
            "介绍一下你自己",
        ]
        test_prompts = [
            test_prompts_en,
            test_prompts_zh,
            test_prompts_en + test_prompts_zh,
        ][args.prompt_lang]
        for idx, prompt in enumerate(test_prompts):
            print(f"\n📝 [text-{idx + 1}]: {prompt}")
            eval_sample(
                model, tokenizer, args, idx, prompt, None, f"text-{idx:02d}.mp3"
            )

    if "1" in modes:
        print(
            "\n\n==================== multi-turn -> {text, audio} ===================="
        )
        multi_turn_tests_zh = [
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                ],
                "prompt": "我想找点事做，你有什么建议吗？",
            },
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                    {"role": "user", "content": "我想找点事做，你有什么建议吗？"},
                    {
                        "role": "assistant",
                        "content": "可以听听音乐或者看看书，放松一下心情。",
                    },
                ],
                "prompt": "好的，那我去照做了，谢谢你",
            },
        ]
        multi_turn_tests_en = [
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ],
                "prompt": "I want to find something to do. Do you have any suggestions?",
            },
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                    {
                        "role": "user",
                        "content": "I want to find something to do. Do you have any suggestions?",
                    },
                    {
                        "role": "assistant",
                        "content": "You can listen to music or read a book to relax a little.",
                    },
                ],
                "prompt": "Okay, I will try that. Thank you.",
            },
        ]
        multi_turn_tests = [
            multi_turn_tests_en,
            multi_turn_tests_zh,
            multi_turn_tests_en + multi_turn_tests_zh,
        ][args.prompt_lang]
        for idx, test in enumerate(multi_turn_tests):
            print(f"\n💬 [multi-{idx + 1}]")
            for msg in test["history"]:
                print(f"   {msg['role']}: {msg['content']}")
            print(f"   user: {test['prompt']}")
            eval_sample(
                model,
                tokenizer,
                args,
                idx,
                test["prompt"],
                None,
                f"multi-{idx:02d}.mp3",
                history=test["history"],
            )

    if "2" in modes:
        print("\n\n==================== audio -> {text, audio} ====================")
        audio_files_en = sorted(
            [
                f
                for f in os.listdir(args.audio_dir)
                if f.startswith("audio-en-") and f.lower().endswith((".mp3", ".wav"))
            ]
        )
        audio_files_zh = sorted(
            [
                f
                for f in os.listdir(args.audio_dir)
                if f.startswith("audio-zh-") and f.lower().endswith((".mp3", ".wav"))
            ]
        )
        audio_files = [audio_files_en, audio_files_zh, audio_files_en + audio_files_zh][
            args.prompt_lang
        ]
        for idx, audio_file in enumerate(audio_files):
            print(f"\n🎤 [audio-{idx + 1}]: {audio_file}")
            #mel=(n_mels, time_frames)=【梅尔滤波器的数量，时间帧数】
            mel, valid_len = OmniDataset.process_audio(
                os.path.join(args.audio_dir, audio_file), model.audio_processor
            )
            audio_inputs = mel.unsqueeze(0).to(args.device)#形状变为(1, n_mels, time_frames)
            audio_lens = torch.tensor([valid_len], device=args.device)
            audio_token_len = valid_len or 1#确保音频 token 长度至少为 1
            prompt = model.config.audio_special_token * audio_token_len
            eval_sample(
                model,
                tokenizer,
                args,
                idx,
                prompt,
                audio_inputs,
                f"audio-{idx:02d}-{os.path.splitext(audio_file)[0]}.mp3",
                audio_lens=audio_lens,
            )

    if "3" in modes:
        print(
            "\n\n==================== clone voice -> {text, audio} ===================="
        )
        clone_prompts_en = [
            "Hello, please introduce yourself.",
            "What's the weather like today?",
            "Tell me a joke.",
        ]
        clone_prompts_zh = [
            "你好，请介绍一下你自己。",
            "今天天气怎么样？",
            "给我讲个笑话吧。",
        ]
        clone_prompts = [
            clone_prompts_en,
            clone_prompts_zh,
            clone_prompts_en + clone_prompts_zh,
        ][args.prompt_lang]
        voices_pt = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model",
            "speaker",
            "voices_unseen.pt",
        )
        voices = [("default", None, None)]
        if os.path.exists(voices_pt):
            voice_data = torch.load(voices_pt, map_location=args.device)
            for speaker, v in sorted(voice_data.items()):
                # rc形状 -> (1, 8, T)，并移至指定设备
                # se=Speaker Embedding，rc=Reference Codes
                rc = v["ref_codes"].unsqueeze(0).to(args.device)
                se = (
                    v["spk_emb"].half().unsqueeze(0).to(args.device)
                    if "spk_emb" in v
                    else None
                )
                voices.append((speaker, rc, se))
        for speaker, rc, se in voices:
            info = (
                f"ref_codes: {rc.shape[2]} frames, spk_emb: {'+' if se is not None else '-'}"
                if rc is not None
                else ("spk_emb only" if se is not None else "default")
            )
            print(f"\n🎵 [clone: {speaker}] {info}")
            for idx, prompt in enumerate(clone_prompts):
                print(f"  📝 [text-{idx + 1}]: {prompt}")
                history = [
                    {
                        "role": "system",
                        "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。请尽量详细地回答，给出有价值的信息。",
                    }
                ]
                eval_sample(
                    model,
                    tokenizer,
                    args,
                    idx,
                    prompt,
                    None,
                    f"clone-{speaker}-{idx:02d}.mp3",
                    ref_codes=rc,
                    history=history,
                    spk_emb=se,
                )

    if "4" in modes:
        print("\n\n==================== image -> {text, audio} ====================")
        image_files = sorted(
            [
                f
                for f in os.listdir(args.image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        for idx, image_file in enumerate(image_files):
            print(f"\n🖼️ [image-{idx + 1}]: {image_file}")
            image = Image.open(os.path.join(args.image_dir, image_file)).convert("RGB")
            pixel_values = {
                k: v.to(args.device)
                for k, v in model.vision_processor(
                    images=image, return_tensors="pt"
                ).items()
            }
            prompts = [
                ["Please describe this image."],
                ["请描述这张图片"],
                ["Please describe this image.", "请描述这张图片"],
            ][args.prompt_lang]
            for lang_idx, prompt_text in enumerate(prompts):
                prompt = (
                    prompt_text
                    + "\n\n"
                    + model.config.image_special_token * model.config.image_token_len
                )
                eval_sample(
                    model,
                    tokenizer,
                    args,
                    idx,
                    prompt,
                    None,
                    f"image-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3",
                    pixel_values=pixel_values,
                )

    if "5" in modes:
        print(
            "\n\n==================== text+audio+image -> {text, audio} ===================="
        )
        img_audio_files = sorted(
            [
                f
                for f in os.listdir(args.audio_dir)
                if f.startswith("img-") and f.lower().endswith((".mp3", ".wav"))
            ]
        )
        image_files = sorted(
            [
                f
                for f in os.listdir(args.image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        text_hints = [
            ["Please answer me: "],
            ["请回答我："],
            ["Please answer me: ", "请回答我："],
        ][args.prompt_lang]
        for idx, image_file in enumerate(image_files):
            audio_file = random.choice(img_audio_files)
            image = Image.open(os.path.join(args.image_dir, image_file)).convert("RGB")
            pixel_values = {
                k: v.to(args.device)
                for k, v in model.vision_processor(
                    images=image, return_tensors="pt"
                ).items()
            }
            for lang_idx, text_hint in enumerate(text_hints):
                print(
                    f"\n🌀 [mix-{idx + 1}-{lang_idx}]: {text_hint} | {audio_file} | {image_file}"
                )
                mel, valid_len = OmniDataset.process_audio(
                    os.path.join(args.audio_dir, audio_file), model.audio_processor
                )
                audio_inputs = mel.unsqueeze(0).to(args.device)
                audio_lens = torch.tensor([valid_len], device=args.device)
                audio_token_len = valid_len or 1
                prompt = (
                    text_hint
                    + model.config.audio_special_token * audio_token_len
                    + "\n\n"
                    + model.config.image_special_token * model.config.image_token_len
                )
                eval_sample(
                    model,
                    tokenizer,
                    args,
                    idx,
                    prompt,
                    audio_inputs,
                    f"mix-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3",
                    pixel_values=pixel_values,
                    audio_lens=audio_lens,
                )


if __name__ == "__main__":
    main()
