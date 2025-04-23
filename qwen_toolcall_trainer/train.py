# qwen_toolcall_trainer/train.py
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

def parse_log_to_chatml(input_log_path, output_jsonl_path):
    with open(input_log_path, "r", encoding="utf-8") as f:
        raw_logs = json.load(f)

    with open(output_jsonl_path, "w", encoding="utf-8") as out:
        for log in raw_logs:
            user_query = log["user"]
            tool_name = log["tool_name"]
            arguments = log["arguments"]

            formatted = {
                "input": (
                    "<|im_start|>system\n你是一个可以调用工具的智能助手<|im_end|>\n"
                    f"<|im_start|>user\n{user_query}<|im_end|>\n"
                    f"<|im_start|>assistant\n<function_call>{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(arguments, ensure_ascii=False)}}}</function_call><|im_end|>"
                )
            }
            out.write(json.dumps(formatted, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parse_log_to_chatml(config["log_path"], config["train_data_path"])

    dataset = load_dataset("json", data_files=config["train_data_path"])['train']

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], trust_remote_code=True)

    def tokenize(example):
        return tokenizer(example['input'], truncation=True, padding='max_length', max_length=config["max_length"])

    tokenized_dataset = dataset.map(tokenize)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
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

if __name__ == "__main__":
    main()
