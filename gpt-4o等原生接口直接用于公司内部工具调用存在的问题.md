# gpt-4o等原生接口直接用于公司内部工具调用存在的问题

---

query: 我想今天去看电影，如果天气不错的话，推荐一家附近评分高的电影院。

提供的可选工具列表为:

```json
tools = [
  {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "城市名称"},
        "date": {"type": "string", "description": "日期"},

      },
      "required": ["city","date"]
    }
  },
  {
    "name": "search_cinemas",
    "description": "搜索评分高的电影院",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "sort_by": {"type": "string", "enum": ["rating", "distance"]},
        "open_now": {"type": "boolean"}
      },
      "required": ["location", "sort_by", "open_now"]
    }
  }
]
```

从用户的query中无法解析出city,location,open_now等参数，但由于OpenAI function-calling 体系中的一个「默认行为陷阱」：

✅ 它知道参数必须被填满

❌ 但在没有上下文时，会“自作主张”生成 "你的城市" 这类猜测或占位字符串，而不是让你来补全或者直接补问。

即使你在prompt反复强调不要自行填充参数，它仍会生成假的参数。

比如上面的例子，用4o模型执行后，LLM的输出为
```json
[
  {'role': 'system','content': '你是一个能根据用户需求主动规划任务步骤、合理调用多个工具的智能助手。你需要判断是否先调用一个工具，再根据结果决定是否调用下一个工具。'},
  {'role': 'user', 'content': '我想今天去看电影，如果天气不错的话，推荐一家附近评分高的电影院。'},
  {'role': 'assistant', 'content': None,'tool_calls': [
    {'id': 'call_81ShbBPDOJCF2ZWTDYIvtQQs','type': 'function','function': {'name': 'get_weather', 'arguments': '{"city": "深圳", "date": "2023-10-22"}'}},
    {'id': 'call_O0J5AedofEjAzuMZ99fgTRvO', 'type': 'function', 'function': {'name': 'search_cinemas', 'arguments': '{"location": "深圳", "sort_by": "rating", "open_now": true}'}}
  ]
  }
]
```

可以看到，在LLM的输出中，city：深圳，date：2023-10-22都是LLM编造的参数，实际用户的城市为北京，date是2024-04-27。

## 为什么会这样？

OpenAI 的模型行为（GPT-4 / GPT-4o）：

知道 tools 的 required 参数（你传了 schema）；

缺少上下文也会强行生成参数（即使值不可信）；

不会在工具调用阶段主动「空缺字段」或「触发补问」；

它宁愿猜一个"假的"值，也不会让 arguments 留空（如 null）；

## 解决方案：

让模型只负责提取“意图 + 所需参数名”

工程师来决定每个参数是否有值，并构造 tool_call

实现方式：两阶段意图识别 + Tool 补全

你在 prompt 或 system message 中告诉它：

如果你无法确定某个参数值，不要编造，直接返回 null 或不要包含该字段。

让它返回格式类似：

```json
{
  "tool": "get_weather",
  "args": {
    "city": null,
    "date": "2025-04-25"
  }
}
```

示例prompt：

```json
你是一个负责规划 tool 调用的助手。请从用户的请求中提取出：
- 调用哪个工具（工具名）
- 哪些参数是明确的（提供值）
- 哪些参数是缺失的（返回 null）

不要胡乱猜测参数值，如果值没有被用户明确提到，就返回 null。
```

第二步：在系统中你来处理补全逻辑
```python
def construct_final_tool_call(parsed, user_context, fallback):
    args = {}
    for k, v in parsed["args"].items():
        if v is not None:
            args[k] = v
        elif k in user_context:
            args[k] = user_context[k]
        elif k in fallback:
            args[k] = fallback[k]
        else:
            # 缺失且无法补全，可提示用户
            raise ValueError(f"参数 {k} 缺失，无法补全")
    
    return {
        "name": parsed["tool"],
        "arguments": args
    }
```






