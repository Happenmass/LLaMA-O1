from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from typing import Dict, List, Union

# 1. 加载tokenizer和模型
def load_tokenizer_and_model(model_name: str, tokenizer_path:str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        pad_token='<|endoftext|>'  # Qwen使用这个作为pad token
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用float16来节省显存
    )
    
    # 确保model知道pad_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model

# 2. 数据预处理函数
def preprocess_function(examples: Dict[str, List[str]], tokenizer, max_length: int = 1024):
    # 将文本转换为token ids
    tokenized_inputs = tokenizer(
        examples["text_cn"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    return tokenized_inputs

# 3. 主训练函数
def train_qwen(
    model_name: str,
    tokenizer_name: str,
    dataset_name: str,
    output_dir: str,
    batch_size: int = 4,
    save_steps: int = 1000,
    learning_rate: float = 2e-5
):
    # 加载tokenizer和模型
    tokenizer, model = load_tokenizer_and_model(model_name, tokenizer_name)
    
    # 加载数据集
    dataset = load_from_disk(dataset_name)
    if "train" not in dataset:
        raise ValueError(f"Dataset {dataset_name} does not have a train split")
    
    # 预处理数据集
    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=10,
        num_train_epochs=1,
        fp16=True,  # 使用混合精度训练
        gradient_accumulation_steps=4,  # 梯度累积
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard"  # 避免输出额外信息
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Qwen是自回归模型，不使用MLM
    )
    
    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型和tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer

# 使用示例
if __name__ == "__main__":
    # 配置参数
    MODEL_NAME = "/home/ecs-user/nas_checkpoint/qwen-7"  # 或其他Qwen模型版本
    TOKENIZER_NAME = "/home/ecs-user/code/happen/PPO/LLaMA-O1/custom_utils/expanded_qwen-7"  # 或其他Qwen模型版本
    DATASET_NAME = "/home/ecs-user/nas_training_data/OpenLongCoT-Pretrain_hf_data_translate_clean"  # 替换为你的数据集名称
    OUTPUT_DIR = "./qwen_pretrained"
    
    # 开始训练``
    trainer = train_qwen(
        model_name=MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME,
        dataset_name=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        batch_size=1,
        save_steps=1000,
        learning_rate=2e-5
    )