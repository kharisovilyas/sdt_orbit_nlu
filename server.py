#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import yaml
import torch
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ... (проверки версий и настройки логирования остаются без изменений) ...

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Orbit-NLU Parser")

class ParseRequest(BaseModel):
    text: str

class ParseResponse(BaseModel):
    filters: dict
    raw: str

def load_cfg():
    """Чтение конфигурационного файла YAML."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении config.yaml: {e}")
        raise

def build_prompt(system_prompt, user):
    """Формирование промта для модели."""
    return f"{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**"

def fix_json(raw_text: str) -> dict:
    """
    Извлекает и парсит первый найденный JSON-объект из строки.
    """
    start_index = raw_text.find('{')
    end_index = raw_text.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        logger.warning(f"Не удалось найти JSON-объект в строке: {raw_text}")
        return {}

    json_str = raw_text[start_index : end_index + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON: {e}. Содержимое: {json_str}")
        return {}

# Загрузка модели и токенизатора при старте сервера
cfg = load_cfg()
base = cfg["model_name"]
out = cfg["output_dir"]
tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
logger.info(f"Токенизатор загружен: {base}")

quant = BitsAndBytesConfig(
    load_in_8bit=cfg.get("load_8bit", True),
    bnb_8bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base,
    device_map="auto",
    quantization_config=quant,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, out)
model.eval()
logger.info(f"Модель с LoRA-адаптером загружена: {out}")

@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    try:
        logger.info(f"Получен запрос: {req.text}")
        prompt = build_prompt(cfg["system_prompt"], req.text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # Используем стабильные параметры генерации
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
            
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw = text.split("**Ответ:**")[-1].strip()
        
        # Используем улучшенную функцию парсинга
        filters = fix_json(raw)
        
        if not filters:
            logger.warning("Не удалось распарсить JSON, возвращаем пустой словарь.")
            
        logger.info(f"Сгенерированные фильтры: {filters}")
        return {"filters": filters, "raw": raw}
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)