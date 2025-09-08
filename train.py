#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Дообучение LLM с LoRA для маппинга русских промтов на JSON-фильтры (SFT).
Оптимизировано для астрономических и баллистических данных малых и больших КА.
Использует transformers, datasets, peft и trl (SFTTrainer).
"""

import os
import json
import yaml
import random
import logging
from typing import Dict
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from sklearn.metrics import precision_score, recall_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def read_yaml(path: str) -> Dict:
    """Чтение конфигурационного файла YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении конфигурации {path}: {e}")
        raise

def format_example(example: Dict, system_prompt: str, prompt_template: str) -> Dict:
    """Форматирование одного примера для обучения."""
    user = example["prompt"]
    target_json = json.dumps(example["filters"], ensure_ascii=False)
    text = prompt_template.format(system_prompt=system_prompt, user=user)
    return {"text": text, "labels": target_json}

def build_dataset(path: str, system_prompt: str, prompt_template: str, split_ratio: float = 0.9) -> tuple:
    """Построение тренировочной и валидационной выборок из JSONL."""
    try:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line.strip())
                    data.append(format_example(ex, system_prompt, prompt_template))

        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        train_data = Dataset.from_list(data[:split_idx])
        val_data = Dataset.from_list(data[split_idx:]) if split_idx < len(data) else None

        logger.info(f"Загружено {len(train_data)} тренировочных примеров и {len(val_data or [])} валидационных.")
        return train_data, val_data
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета из {path}: {e}")
        raise

def compute_metrics(eval_pred):
    """Кастомная метрика для оценки качества JSON-ответов."""
    predictions, labels = eval_pred
    pred_str = [pred.decode("utf-8") for pred in predictions]
    label_str = [label.decode("utf-8") for label in labels]
    exact_matches = [1 if pred == label else 0 for pred, label in zip(pred_str, label_str)]
    precision = precision_score(exact_matches, [1] * len(exact_matches), zero_division=0)
    recall = recall_score(exact_matches, [1] * len(exact_matches), zero_division=0)
    return {"exact_match_precision": precision, "exact_match_recall": recall}

def main():
    try:
        # Загрузка конфигурации
        cfg = read_yaml("config.yaml")
        logger.info("Конфигурация успешно загружена.")

        model_name = cfg["model_name"]
        load_8bit = cfg.get("load_8bit", False)

        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Токенизатор загружен: {model_name}")

        # Квантование
        quant_config = None
        if load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )

        # Модель
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if load_8bit else torch.float16
        )
        if load_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info(f"Модель загружена: {model_name}")

        # LoRA
        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        logger.info("LoRA-конфигурация применена.")

        # Датасеты
        system_prompt = cfg["system_prompt"]
        prompt_template = cfg["prompt_template"]
        val_path = cfg.get("val_dataset_path")
        if val_path and os.path.exists(val_path):
            train_data = Dataset.from_list(build_dataset(cfg["dataset_path"], system_prompt, prompt_template)[0])
            val_data = Dataset.from_list(build_dataset(val_path, system_prompt, prompt_template)[0])
        else:
            train_data, val_data = build_dataset(
                cfg["dataset_path"], system_prompt, prompt_template, cfg.get("split_ratio", 0.9)
            )

        # Конфигурация обучения
        training_args = TrainingArguments(
            output_dir=cfg["output_dir"],
            num_train_epochs=cfg["num_train_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            warmup_ratio=cfg["warmup_ratio"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            evaluation_strategy="steps" if val_data else "no",  # тут еще маленький фикс
            eval_steps=cfg["eval_steps"] if val_data else None,
            fp16=not load_8bit,
            bf16=load_8bit,
            report_to="none",
            gradient_checkpointing=True,
            dataloader_num_workers=cfg.get("dataset_num_proc", 1),
        )

        # Инициализация тренера
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            tokenizer=tokenizer,                     # лучше явно передать токенизатор
            compute_metrics=compute_metrics if val_data else None,
            max_seq_length=cfg["max_seq_length"],    # <-- перенес сюда
            dataset_text_field="text",               # нужно указать поле с текстом
        )

        logger.info("Тренер инициализирован.")

        # Обучение
        trainer.train()
        logger.info("Обучение завершено.")

        # Сохранение
        trainer.model.save_pretrained(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        logger.info(f"Модель и токенизатор сохранены в {cfg['output_dir']}")

    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise

if __name__ == "__main__":
    main()