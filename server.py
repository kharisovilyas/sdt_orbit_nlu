#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI-сервер для парсинга русских промтов о спутниках в JSON-фильтры.
Оптимизировано для астрономических и баллистических данных малых и больших КА.
Требуемые зависимости: transformers>=4.44.0, peft>=0.12.0, fastapi>=0.111.0,
uvicorn>=0.30.0, pyyaml>=6.0.1, torch.
"""

import os
import re
import json
import ast
import yaml
import torch
import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Проверка версий библиотек (информативная) ---
try:
    import transformers as _transformers
    import peft as _peft
    import fastapi as _fastapi
    import uvicorn as _uvicorn
    assert _transformers.__version__ >= "4.44.0", "Требуется transformers>=4.44.0"
    assert _peft.__version__ >= "0.12.0", "Требуется peft>=0.12.0"
    assert _fastapi.__version__ >= "0.111.0", "Требуется fastapi>=0.111.0"
    assert _uvicorn.__version__ >= "0.30.0", "Требуется uvicorn>=0.30.0"
except ImportError as e:
    raise ImportError(f"Ошибка импорта библиотеки: {e}. Установите зависимости.")
except AssertionError as e:
    raise AssertionError(f"Несовместимая версия библиотеки: {e}")

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orbit-nlu")

app = FastAPI(title="Orbit-NLU Parser")

# --- Pydantic модели ---
class ParseRequest(BaseModel):
    text: str

class ParseResponse(BaseModel):
    filters: Dict[str, Any]
    raw: str

# --- Конфиг ---
def load_cfg() -> dict:
    """Чтение конфигурационного файла YAML."""
    cfg_path = os.environ.get("ORBIT_CFG", "config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            logger.info(f"Конфигурация загружена из {cfg_path}")
            return cfg
    except FileNotFoundError:
        logger.error(f"Файл конфигурации {cfg_path} не найден.")
        raise
    except Exception as e:
        logger.error(f"Ошибка при чтении {cfg_path}: {e}")
        raise

# --- Промпт билдер (жёсткое требование JSON) ---
def build_prompt(system_prompt: str, user: str) -> str:
    """
    Формирование промта. Просим модель вернуть строго валидный JSON,
    начинающийся с '{' и заканчивающийся '}', без пояснений.
    """
    strict_instruction = (
        "\n\nВОТ ВАЖНОЕ ТРЕБОВАНИЕ: Ответ должен быть строго в формате JSON. "
        "Выдайте только JSON-объект, начинающийся с '{' и заканчивающийся '}'. "
        "Никаких пояснений, ни одного дополнительного текста. "
        "Все ключи должны быть в двойных кавычках."
    )
    return f"{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**{strict_instruction}\n"

# --- Улучшенная функция восстановления JSON ---
def _clean_artifacts(s: str) -> str:
    """Удаляет известные артефакты и повторяющиеся фрагменты."""
    # Уберём повторяющиеся фрагменты вида `"}`: "1"}"` и похожие шумы
    s = re.sub(r'("{0,1}\}\":\s*"{0,1}1"{0,1}\}\s*")+', '', s)
    # Удалим последовательные повторяющиеся закрывающие кавычки/скобки с пробелами
    s = re.sub(r'["\']{2,}', '"', s)
    # Нормализуем Windows-переносы строк
    s = s.replace('\r\n', '\n')
    # Удаляем управляющие символы, кроме таба и новой строки
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    return s

def _extract_json_by_regex(s: str) -> str:
    """Ищет самый большой по длине вхождению JSON-подстроку {...} в тексте."""
    matches = re.findall(r'\{[\s\S]*\}', s)
    if not matches:
        return ""
    # Выберем самый длинный матч (чаще всего это полный JSON)
    return max(matches, key=len)

def _try_fix_trailing_commas(s: str) -> str:
    """Удаляет лишние запятые перед закрывающей скобкой."""
    # { "a": 1, } -> { "a": 1 }
    s = re.sub(r',\s*(\}|])', r'\1', s)
    return s

def fix_json(raw: str) -> dict:
    """
    Попытка восстановить и распарсить JSON-объект из текста.
    Возвращает dict или пустой dict при неудаче.
    """
    if not raw or not isinstance(raw, str):
        return {}

    text = raw.strip()
    text = _clean_artifacts(text)

    # Попробуем классический json.loads сначала
    try:
        return json.loads(text)
    except Exception:
        pass

    # Попробуем найти {...} в тексте
    candidate = _extract_json_by_regex(text)
    if candidate:
        candidate = _try_fix_trailing_commas(candidate)
        try:
            return json.loads(candidate)
        except Exception as e:
            logger.debug(f"json.loads(candidate) failed: {e}")

    # Если фигурных скобок нет, попробуем обернуть в {}
    # и заменить одиночные кавычки на двойные (с осторожностью)
    no_braces = text
    # Удалим повторяющиеся новые строки и обрежем лишний текст
    no_braces = re.sub(r'\n{2,}', '\n', no_braces).strip()

    # Если строка выглядит как список пар "key": "value", обернём
    if re.search(r'"\w+"\s*:\s*', no_braces) or re.search(r'\w+\s*:\s*"', no_braces):
        candidate2 = no_braces
        # Попытка обернуть отсутствующие фигурные скобки
        if not candidate2.startswith('{'):
            candidate2 = '{' + candidate2
        if not candidate2.endswith('}'):
            candidate2 = candidate2 + '}'
        candidate2 = _try_fix_trailing_commas(candidate2)
        # Приведём одиночные кавычки к двойным (бережно)
        candidate2 = re.sub(r"(?<!\\)'", '"', candidate2)
        # Удалим повторяющиеся запятые
        candidate2 = re.sub(r',\s*,+', ',', candidate2)
        # Удалим случайные вложенные невалидные части
        candidate2 = re.sub(r'\}\s*,\s*\}', '}}', candidate2)

        try:
            return json.loads(candidate2)
        except Exception as e:
            logger.debug(f"json.loads(candidate2) failed: {e}")

    # Как запасной вариант — попытаемся через ast.literal_eval после небольшого препроцессинга
    # Преобразуем true/false/null в Python-аналоги
    alt = text
    alt = re.sub(r'\bnull\b', 'None', alt, flags=re.IGNORECASE)
    alt = re.sub(r'\btrue\b', 'True', alt, flags=re.IGNORECASE)
    alt = re.sub(r'\bfalse\b', 'False', alt, flags=re.IGNORECASE)
    alt = _try_fix_trailing_commas(alt)
    # Если нет фигурных скобок — обернём
    if not alt.strip().startswith('{') and ':' in alt:
        alt = '{' + alt + '}'
    try:
        parsed = ast.literal_eval(alt)
        # ast.literal_eval может вернуть tuple/list, ожидаем dict
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.debug(f"ast.literal_eval failed: {e}")

    logger.warning("Не удалось восстановить валидный JSON из ответа модели.")
    return {}

# --- Загрузка конфигурации и модели ---
cfg = load_cfg()
base = cfg.get("model_name")
out = cfg.get("output_dir")
if not base or not out:
    raise RuntimeError("config.yaml должен содержать ключи 'model_name' и 'output_dir'")

# Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Токенизатор загружен: {base}")
except Exception as e:
    logger.error(f"Не удалось загрузить токенизатор: {e}")
    raise

# Model (с учётом 8-bit конфигурации, если указана)
try:
    quant_cfg = BitsAndBytesConfig(
        load_in_8bit=bool(cfg.get("load_8bit", True)),
        bnb_8bit_compute_dtype=getattr(torch, cfg.get("bnb_8bit_compute_dtype", "bfloat16"))
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    )
    model = PeftModel.from_pretrained(model, out)
    model.eval()
    logger.info(f"Модель и LoRA-адаптер загружены: {out}")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели/адаптера: {e}")
    raise

# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    try:
        logger.info("Получен parse-запрос.")
        prompt = build_prompt(cfg.get("system_prompt", ""), req.text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(next(model.parameters()).device)
        # Генерация: более жёсткие параметры чтобы снизить "галлюцинации"
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            do_sample=bool(cfg.get("do_sample", False)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_k=int(cfg.get("top_k", 50)),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        with torch.no_grad():
            gen_ids = model.generate(**gen_kwargs)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Отделим часть после маркера ответа
        raw = text.split("**Ответ:**")[-1].strip() if "**Ответ:**" in text else text.strip()
        raw_clean = raw
        # Очистим часто встречающийся мусор в конце
        raw_clean = re.sub(r'(\n\s*)+$', '', raw_clean)
        filters = fix_json(raw_clean)
        if not filters:
            logger.warning("Не удалось распарсить JSON — возвращаем пустой словарь вместе с сырой строкой.")
        logger.info(f"Распарсено полей: {len(filters)}")
        return {"filters": filters, "raw": raw_clean}
    except Exception as e:
        logger.exception("Ошибка при обработке parse-запроса")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

# --- Запуск приложения при старте файла напрямую ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)))
