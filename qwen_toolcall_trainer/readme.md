👌 已完成封装，以下是你的一键式训练流程：

---

## 📁 目录结构建议
```
qwen_toolcall_trainer/
├── train.py               # 主训练入口
├── config.yaml            # 训练配置
├── toolcall_logs.json     # 你的原始日志（从调用接口日志生成）
└── qwen_toolcall_train.jsonl   # 自动生成的 ChatML 格式训练数据
```

---

## 🧾 config.yaml 示例配置

```yaml
log_path: "toolcall_logs.json"
train_data_path: "qwen_toolcall_train.jsonl"
model_name_or_path: "Qwen/Qwen-7B-Chat"
output_dir: "./checkpoints/qwen-toolcall"
batch_size: 2
num_train_epochs: 3
save_steps: 100
logging_steps: 20
max_length: 1024
```

---

## 📑 toolcall_logs.json 示例格式（源数据）

```json
[
  {
    "user": "查一下上海的天气",
    "tool_name": "get_weather",
    "arguments": {
      "city": "上海"
    }
  },
  {
    "user": "我想知道故宫的位置",
    "tool_name": "map_lookup",
    "arguments": {
      "place": "故宫"
    }
  }
]
```

---

## 🚀 启动训练

```bash
cd qwen_toolcall_trainer
python train.py --config config.yaml
```

---

如果你希望支持 **多轮对话、多个函数调用、嵌套结构、异常处理样例** 等复杂功能，我们也可以扩展数据构造逻辑。需要我为你提供一个更复杂的样例构造器或评估指标支持（如 Exact Match 工具调用准确率）也可以继续说。
