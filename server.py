#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI-сервер для парсинга русских промтов о спутниках в JSON-фильтры.
Оптимизировано для астрономических и баллистических данных малых и больших КА.
Совместимо с transformers>=4.44.0, peft>=0.12.0, fastapi>=0.111.0, uvicorn>=0.30.0, pyyaml>=6.0.1.
"""

import os
import json
import yaml
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Проверка версий библиотек
try:
    import transformers
    import peft
    import fastapi
    import uvicorn
    import yaml
    assert transformers.__version__ >= "4.44.0", "Требуется transformers>=4.44.0"
    assert peft.__version__ >= "0.12.0", "Требуется peft>=0.12.0"
    assert fastapi.__version__ >= "0.111.0", "Требуется fastapi>=0.111.0"
    assert uvicorn.__version__ >= "0.30.0", "Требуется uvicorn>=0.30.0"
except ImportError as e:
    raise ImportError(f"Ошибка импорта библиотеки: {e}. Убедитесь, что установлены все зависимости.")
except AssertionError as e:
    raise AssertionError(f"Несовместимая версия библиотеки: {e}")

# Настройка логирования
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

def fix_json(raw: str) -> dict:
    """Попытка исправить некорректный JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Некорректный JSON: {e}. Пытаемся исправить...")
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                logger.error("Не удалось исправить JSON.")
                return {}
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
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_k=50
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw = text.split("**Ответ:**")[-1].strip()
        filters = fix_json(raw)
        if not filters:
            logger.warning("Не удалось распарсить JSON, возвращаем пустой словарь.")
        logger.info(f"Сгенерированные фильтры: {filters}")
        return {"filters": filters, "raw": raw}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)