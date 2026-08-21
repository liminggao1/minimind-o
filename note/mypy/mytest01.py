from transformers import AutoTokenizer
import json


def main():
    tokenizer = AutoTokenizer.from_pretrained("./model")

    print("=" * 80)
    print("【chat_template Jinja2源码片段预览】")
    print("=" * 80)


    # 工具定义：两个函数
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询某地实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "联网搜索获取外部知识",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询词"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # ======================
    # 最复杂全集样例 messages_max_complex
    # 1. system系统提示
    # 2. user提问
    # 3. assistant：reasoning_content独立字段 + 双并行tool_call
    # 4. 连续两条tool返回
    # 5. assistant：content内部自带标签（模板会自动切割解析）
    # 6. user追问
    # 7. assistant普通回答，不带工具
    # 8. user再次追问，结束历史，交给add_generation_prompt继续生成
    # ======================
    messages_max_complex = [
        {"role": "system", "content": "你是擅长使用工具的AI助手，需要外部信息优先调用工具。"},
        {"role": "user", "content": "湖州今天天气怎么样，常住人口有多少？"},
        {
            "role": "assistant",
            "reasoning_content": "用户需要天气和人口，两个都需要联网获取，并行调用两个工具。",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "湖州"}
                    }
                },
                {
                    "function": {
                        "name": "search_web",
                        "arguments": {"query": "湖州常住人口数量"}
                    }
                }
            ]
        },
        {"role": "tool", "content": '{"city":"湖州","temp":27,"weather":"多云"}'},
        {"role": "tool", "content": '{"result":"湖州常住人口约341.3万"}'},
        {
            "role": "assistant",
            "content": "工具拿到结果，整理后回复用户。\n湖州今日多云，气温27℃；常住人口约341.3万。"
        },
        {"role": "user", "content": "这个温度适合户外游玩吗？"},
        {
            "role": "assistant",
            "content": "27度多云很适合出游。\n27℃多云天气比较舒适，适合户外游玩。"
        },
        {"role": "user", "content": "那湖州有什么值得去的景点？"}
    ]

    print("\n" + "=" * 80)
    print("【测试：最大复杂度全集场景】")
    print("=" * 80)
    s_complex = tokenizer.apply_chat_template(
        messages_max_complex,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=True
    )
    print("\n=====渲染后完整prompt=====\n")
    print(s_complex)

    # 转为token id，看送入模型真实序列
    input_ids = tokenizer.apply_chat_template(
        messages_max_complex,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        open_thinking=True
    )
    print("\n" + "=" * 80)
    print(f"总token长度：{len(input_ids)}")
    print("input_ids 前100：")
    print(input_ids[:100])

    # ------------------------------
    # 对比：同一套messages，不传tools，观察：工具相关全部消失
    # ------------------------------
    print("\n" + "=" * 80)
    print("【对比：同样消息，tools=None，关闭工具分支】")
    print("=" * 80)
    s_no_tools = tokenizer.apply_chat_template(
        messages_max_complex,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=True
    )
    print("\n=====不传tools渲染结果（工具XML全部消失）=====\n")
    print(s_no_tools)


if __name__ == "__main__":
    main()