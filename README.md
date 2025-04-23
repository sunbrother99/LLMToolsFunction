# LLMToolsFunction
很好！既然你使用的是 **Qwen 模型**，我们可以将工具调用的 SFT 数据准备为 HuggingFace 格式（也可称为 ChatML 或 Qwen 的训练兼容格式），并配套提供：

- ✅ **数据构造器**：根据 query 和函数名 + 参数生成标准格式
- ✅ **训练数据输出为 JSON/JSONL**
- ✅ **HuggingFace Dataset 格式准备脚本**
- ✅ **训练 pipeline（使用 `transformers` 和 `Trainer`）**

---

## 📦 一、ChatML 格式说明（Qwen 专用）

Qwen 微调通常使用类似如下格式的对话记录（每条数据是完整对话）：

```json
{
  "input": "<|im_start|>system\n你是一个可以调用工具的智能助手<|im_end|>\n<|im_start|>user\n查一下广州天气<|im_end|>\n<|im_start|>assistant\n<function_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"广州\"}}</function_call><|im_end|>"
}
```

Qwen 通过 `<function_call>{...}</function_call>` 标签标识函数调用。

额外的
使用 SFT 训练的数据格式通常为：

📄 常见格式：OpenAI、ChatGLM、Baichuan、Qwen 均支持这种格式
{
  "messages": [
    {"role": "system", "content": "你是一个可以调用内部工具的智能助手"},
    {"role": "user", "content": "帮我查一下广州的天气"},
    {
      "role": "assistant",
      "function_call": {
        "name": "get_weather",
        "arguments": "{ \"city\": \"广州\" }"
      }
    }
  ]
}
最上面的qwen格式是qwen模型的专用格式，如果以openai的接口形式组装，则是【常见格式】这种。

---

## 🧰 二、数据构造器（Python 脚本）

创建一个脚本 `generate_qwen_toolcall_dataset.py`：

```python
import json

# 你可以定义更多样例，或从日志构造
examples = [
    ("查一下北京的天气", "get_weather", {"city": "北京"}),
    ("帮我定位上海的位置", "map_lookup", {"place": "上海"}),
    ("我想搜索一下Qwen模型的用途", "search", {"query": "Qwen模型的用途"}),
]

def build_chatml_entry(query, tool_name, args):
    return {
        "input": (
            "<|im_start|>system\n你是一个可以调用工具的智能助手<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<function_call>{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(args, ensure_ascii=False)}}}</function_call><|im_end|>"
        )
    }

def main():
    output_path = "qwen_toolcall_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for query, tool, args in examples:
            record = build_chatml_entry(query, tool, args)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"数据集已生成：{output_path}")

if __name__ == "__main__":
    main()
```

---

## 📂 三、加载 HuggingFace Dataset 格式（用于训练）

使用 `datasets` 库将生成的 `jsonl` 转为 HuggingFace 格式：

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='qwen_toolcall_train.jsonl')
print(dataset['train'][0])
```

---

## 🔧 四、训练 Pipeline 脚本（transformers + Qwen）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

model_path = "Qwen/Qwen-7B-Chat"  # 或你的本地路径
dataset = load_dataset('json', data_files='qwen_toolcall_train.jsonl')['train']

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

def tokenize(example):
    return tokenizer(example['input'], truncation=True, padding='max_length', max_length=1024)

tokenized_dataset = dataset.map(tokenize)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qwen-toolcall-checkpoints",
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=50,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
```

---

## ✅ 输出样例（训练后效果预期）：

输入用户问句：
```
查一下北京的天气
```

生成回复：
```
<function_call>{"name": "get_weather", "arguments": {"city": "北京"}}</function_call>
```

---

## 🎯 总结

| 模块 | 内容 |
|------|------|
| ✅ 数据格式 | ChatML 格式，使用 `<function_call>` 标签 |
| ✅ 构造器 | 支持从 query + 工具 + 参数合成训练样本 |
| ✅ 训练 | 使用 HuggingFace `Trainer` 完成微调 |
| ✅ 目标 | 提高 Qwen 在 function calling 场景下的准确性和鲁棒性 |

---

需要我帮你：
- 自动从接口日志构造大批量数据
- 封装为 `train.py + config.yaml` 可一键训练形式
- 支持多轮调用、函数嵌套等复杂场景

可以继续告诉我，我可以帮你自动化整理和部署微调系统。是否也需要验证代码是否能在你现在的服务器环境下跑通？
