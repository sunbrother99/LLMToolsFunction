"""
  你提供一个 query 列表，GPT-4 输出：

需要调用的工具名称

对应参数（JSON 格式）

可选：完整 tool chain（多工具调用顺序）

最后将这些写入一个标准的 log JSON 文件，用于后续 SFT 训练。

  输出格式（每条数据）：

  {
  "user": "查一下北京的天气",
  "tool_name": "get_weather",
  "arguments": {
    "city": "北京"
  }
}

或支持多工具调用时：

{
  "user": "我想查天气然后找家附近的评分高的电影院",
  "tool_name": "get_weather",
  "arguments": { "city": "广州" },
  "chain": [
    { "name": "get_weather", "arguments": { "city": "广州" } },
    { "name": "search_cinemas", "arguments": { "location": "广州", "sort_by": "rating" } }
  ]
}

"""

import openai
import json

openai.api_key = "YOUR_OPENAI_API_KEY"  # 替换为你的 GPT-4 Key

# 示例用户 query 列表
queries = [
    "查一下北京的天气",
    "我想知道今天上海适合出门吗",
    "找个附近评分高的餐厅",
    "如果天气合适，我想看电影，推荐一家影院",
]

# 提示模板
system_prompt = """你是一个负责将用户请求映射为工具调用的助手。
请根据用户的自然语言输入，判断要调用的工具名称和参数。支持如下工具：

- get_weather(city): 查询某个城市的天气
- search_restaurants(location, sort_by): 搜索附近餐厅，sort_by 可为 rating 或 distance
- search_cinemas(location, sort_by): 搜索电影院
- search_hotels(location, stars): 搜索酒店

请输出 JSON 格式的结果，如：
{
  "user": "用户原始输入",
  "tool_name": "get_weather",
  "arguments": {"city": "北京"}
}

如果需要多个步骤（chain of tools），请附加一个 chain 字段（按顺序排列）：
{
  "user": "我想先查天气再查餐厅",
  "tool_name": "get_weather",
  "arguments": {"city": "广州"},
  "chain": [
    {"name": "get_weather", "arguments": {"city": "广州"}},
    {"name": "search_restaurants", "arguments": {"location": "广州", "sort_by": "rating"}}
  ]
}
"""

# 执行 GPT-4 调用
def gpt_generate_tool_logs(queries):
    logs = []
    for q in queries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2
        )
        result = response.choices[0].message["content"]
        try:
            logs.append(json.loads(result))
        except json.JSONDecodeError as e:
            print("解析失败:", q)
            print("返回:", result)
    return logs

# 写入日志文件
def save_logs(logs, path="toolcall_logs.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"保存成功: {path}")

# 运行生成 + 保存
logs = gpt_generate_tool_logs(queries)
save_logs(logs)
