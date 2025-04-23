# qwen_toolcall_trainer/train.py
import json
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split

def parse_log_to_chatml(input_log_path, output_jsonl_path):
    with open(input_log_path, "r", encoding="utf-8") as f:
        raw_logs = json.load(f)

    with open(output_jsonl_path, "w", encoding="utf-8") as out:
        for log in raw_logs:
            user_query = log["user"]
            tool_name = log["tool_name"]
            arguments = log["arguments"]
            chain = log.get("chain", [])

            formatted = {
                "input": (
                    "<|im_start|>system\n你是一个可以调用工具的智能助手<|im_end|>\n"
                    f"<|im_start|>user\n{user_query}<|im_end|>\n"
                    f"<|im_start|>assistant\n<function_call>{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(arguments, ensure_ascii=False)}}}</function_call><|im_end|>"
                ),
                "expected_tool": tool_name,
                "expected_args": arguments,
                "expected_chain": chain
                # 增加正例，反例
                # "rationale": log.get("rationale", ""),
                # "negative": log.get("negative", False)
            }
            out.write(json.dumps(formatted, ensure_ascii=False) + "\n")

def evaluate_model(model, tokenizer, dataset, max_length):
    correct = 0
    total = 0
    partial_match = 0
    failed_cases = []
    tool_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    chain_correct = 0
    chain_total = 0

    for example in dataset:
        input_ids = tokenizer(example['input'], return_tensors='pt', truncation=True, max_length=max_length).input_ids
        output = model.generate(input_ids, max_length=max_length)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        tool_name = example["expected_tool"]
        tool_stats[tool_name]["total"] += 1
        total += 1

        try:
            start = response.index('<function_call>') + len('<function_call>')
            end = response.index('</function_call>')
            content = response[start:end].strip()
            parsed = json.loads(content)

            if parsed["name"] == tool_name:
                if parsed["arguments"] == example["expected_args"]:
                    correct += 1
                    tool_stats[tool_name]["correct"] += 1
                else:
                    pred_args = parsed["arguments"]
                    exp_args = example["expected_args"]
                    matched_keys = set(pred_args.items()) & set(exp_args.items())
                    if len(matched_keys) >= 1:
                        partial_match += 1
                    else:
                        failed_cases.append((example['input'], parsed))
            else:
                failed_cases.append((example['input'], parsed))

            # Chain-level evaluation
            if example.get("expected_chain"):
                chain_total += 1
                match = True
                for idx, step in enumerate(example["expected_chain"]):
                    if idx == 0 and (parsed["name"] != step["name"] or parsed["arguments"] != step["arguments"]):
                        match = False
                        break
                if match:
                    chain_correct += 1

        except Exception as e:
            failed_cases.append((example['input'], str(e)))

    print("\n--- 自动化评估 ---")
    print(f"Tool Call Accuracy: {correct}/{total} = {correct / total:.2%}")
    print(f"Partial Match (some args correct): {partial_match}/{total} = {partial_match / total:.2%}")
    print(f"Total Failures: {len(failed_cases)}")

    print("\n[工具调用准确率分布]:")
    for tool, stats in tool_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f" - {tool}: {stats['correct']}/{stats['total']} = {acc:.2%}")

    if chain_total > 0:
        print(f"\nChain-level Accuracy: {chain_correct}/{chain_total} = {chain_correct / chain_total:.2%}")

    for case in failed_cases[:5]:
        print("\n[Failed Sample]")
        print("Input:", case[0])
        print("Predicted:", case[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parse_log_to_chatml(config["log_path"], config["train_data_path"])

    full_dataset = load_dataset("json", data_files=config["train_data_path"])['train']
    train_test = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], trust_remote_code=True)

    def tokenize(example):
        return tokenizer(example['input'], truncation=True, padding='max_length', max_length=config["max_length"])

    tokenized_train = train_dataset.map(tokenize)
    tokenized_eval = eval_dataset.map(tokenize)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        evaluation_strategy="epoch",
        eval_steps=1,
        fp16=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    evaluate_model(model, tokenizer, eval_dataset, config["max_length"])

if __name__ == "__main__":
    main()
