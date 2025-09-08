import json
import random
from datetime import datetime, timedelta
import requests
import re
import pymorphy2
import sys

morph = pymorphy2.MorphAnalyzer(lang='ru')

# --- Данные для генерации ---
coverages = {
    "Россия": {
        "nom": ["Россия", "российский регион", "территория РФ"],
        "gen": ["России", "российского региона", "территории РФ"],
        "acc": ["Россию", "российский регион", "территорию РФ"],
        "dat": ["России", "российскому региону", "территории РФ"],
        "prep": ["России", "российском регионе", "территории РФ"]
    },
    "Арктика": {
        "nom": ["Арктика", "арктический регион", "северный полюс"],
        "gen": ["Арктики", "арктического региона", "северного полюса"],
        "acc": ["Арктику", "арктический регион", "северный полюс"],
        "dat": ["Арктике", "арктическому региону", "северному полюсу"],
        "prep": ["Арктике", "арктическом регионе", "северном полюсе"]
    },
    "Африка": {
        "nom": ["Африка", "африканский континент", "территория Африки"],
        "gen": ["Африки", "африканского континента", "территории Африки"],
        "acc": ["Африку", "африканский континент", "территорию Африки"],
        "dat": ["Африке", "африканскому континенту", "территории Африки"],
        "prep": ["Африке", "африканском континенте", "территории Африки"]
    },
    "Китай": {
        "nom": ["Китай", "китайский регион", "территория КНР"],
        "gen": ["Китая", "китайского региона", "территории КНР"],
        "acc": ["Китай", "китайский регион", "территорию КНР"],
        "dat": ["Китаю", "китайскому региону", "территории КНР"],
        "prep": ["Китае", "китайском регионе", "территории КНР"]
    },
    "Европа": {
        "nom": ["Европа", "европейский регион", "территория ЕС"],
        "gen": ["Европы", "европейского региона", "территории ЕС"],
        "acc": ["Европу", "европейский регион", "территорию ЕС"],
        "dat": ["Европе", "европейскому региону", "территории ЕС"],
        "prep": ["Европе", "европейском регионе", "территории ЕС"]
    },
    "Южная Америка": {
        "nom": ["Южная Америка", "южноамериканский континент"],
        "gen": ["Южной Америки", "южноамериканского континента"],
        "acc": ["Южную Америку", "южноамериканский континент"],
        "dat": ["Южной Америке", "южноамериканскому континенту"],
        "prep": ["Южной Америке", "южноамериканском континенте"]
    }
}

altitudes = {
    "<600 км": ["<600 км", "менее 600 километров", "ниже 600 км", "высота до 600 км"],
    "<1000 км": ["<1000 км", "менее 1000 километров", "ниже 1000 км", "высота до 1000 км"],
    "500-800 км": ["500-800 км", "от 500 до 800 километров", "в диапазоне 500-800 км", "высота 500-800 км"],
    "~800 км": ["~800 км", "около 800 километров", "приблизительно 800 км", "высота ~800 км"],
    "700-900 км": ["700-900 км", "от 700 до 900 километров", "в диапазоне 700-900 км", "высота 700-900 км"],
    "~36000 км": ["~36000 км", "около 36000 километров", "приблизительно 36000 км", "высота ~36000 км"],
    "2000-20000 км": ["2000-20000 км", "от 2000 до 20000 километров", "в диапазоне 2000-20000 км"],
    "400-40000 км": ["400-40000 км", "с перигеем ~400 км и апогеем ~40000 км"]
}

orbit_types = {
    "LEO": ["низкая околоземная орбита", "низкую околоземную орбиту", "на низкой околоземной орбите"],
    "MEO": ["средняя околоземная орбита", "среднюю околоземную орбиту", "на средней околоземной орбите"],
    "GEO": ["геостационарная орбита", "геостационарную орбиту", "на геостационарной орбите"],
    "SSO": ["солнечно-синхронная орбита", "солнечно-синхронную орбиту", "на солнечно-синхронной орбите"],
    "Molniya": ["орбита Молния", "орбиту Молния", "на орбите Молния", "орбита типа Молния"],
    "HEO": ["высокая эллиптическая орбита", "высокую эллиптическую орбиту", "на высокой эллиптической орбите"]
}

statuses = {
    "активен": ["активен", "активный", "работающий", "функционирующий", "в рабочем состоянии"],
    "неактивен": ["неактивен", "неактивный", "неработающий", "нефункционирующий", "вышедший из строя"]
}

form_factors = {
    "1U": ["1U", "кубсат 1U", "формат 1U"], "2U": ["2U", "кубсат 2U", "формат 2U"],
    "3U": ["3U", "кубсат 3U", "формат 3U"], "6U": ["6U", "кубсат 6U", "формат 6U"],
    "12U": ["12U", "кубсат 12U", "формат 12U"], "24U": ["24U", "кубсат 24U", "формат 24U"],
    "36U": ["36U", "кубсат 36U", "формат 36U"], "48U": ["48U", "кубсат 48U", "формат 48U"]
}

valid_masses_for_form_factor = {
    "1U": (1, 2), "2U": (2, 4), "3U": (4, 7), "6U": (8, 12),
    "12U": (20, 30), "24U": (40, 50), "36U": (60, 80), "48U": (80, 100)
}

valid_altitudes_for_orbit = {
    "LEO": ["<600 км", "<1000 км", "500-800 км", "~800 км", "700-900 км"],
    "MEO": ["2000-20000 км"],
    "GEO": ["~36000 км"],
    "SSO": ["<600 км", "<1000 км", "500-800 км", "~800 км", "700-900 км"],
    "Molniya": ["400-40000 км"],
    "HEO": ["400-40000 км"]
}

request_synonyms = [
    "Подбери", "Найди", "Выведи", "Покажи", "Ищи", "Скинь", "Какие есть", "Нужны", "Требуется",
    "Выбери", "Предоставь", "Обеспечь", "Дай", "Предложи", "Подскажи"
]

numbers_words_nom = {
    1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять", 6: "шесть", 7: "семь",
    8: "восемь", 9: "девять", 10: "десять", 11: "одиннадцать", 12: "двенадцать",
    13: "тринадцать", 14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать",
    17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать", 20: "двадцать"
}
numbers_words_acc = numbers_words_nom
numbers_words_gen = {
    1: "одного", 2: "двух", 3: "трех", 4: "четырех", 5: "пяти", 6: "шести", 7: "семи",
    8: "восьми", 9: "девяти", 10: "десяти", 11: "одиннадцати", 12: "двенадцати",
    13: "тринадцати", 14: "четырнадцати", 15: "пятнадцати", 16: "шестнадцати",
    17: "семнадцати", 18: "восемнадцати", 19: "девятнадцати", 20: "двадцати"
}
numbers_words_dat = {
    1: "одному", 2: "двум", 3: "трем", 4: "четырем", 5: "пяти", 6: "шести", 7: "семи",
    8: "восьми", 9: "девяти", 10: "десяти", 11: "одиннадцати", 12: "двенадцати",
    13: "тринадцати", 14: "четырнадцати", 15: "пятнадцати", 16: "шестнадцати",
    17: "семнадцати", 18: "восемнадцати", 19: "девятнадцати", 20: "двадцати"
}
numbers_words_prep = numbers_words_gen

templates = [
    # утверждения (False)
    ("Найди {satelliteTextAcc} на {orbitType} высотой {altitude} для наблюдения {coverageGen}.", {"number": None, "orbitType": None, "altitude": None, "coverage": None}, False),
    ("Пожалуйста, подбери {satelliteTextAcc} для мониторинга {coverageGen} с массой {mass}.", {"number": None, "coverage": None, "mass": None}, False),
    ("Запрос: вывести {satelliteTextAcc} с форм-фактором {formFactor}.", {"number": None, "formFactor": None}, False),
    ("Обеспечь список {satelliteTextGen} с массой {mass} и статусом {status}.", {"number": None, "mass": None, "status": None}, False),
    ("Нужна информация о {satelliteTextPrep} с покрытием {coverageGen}.", {"number": None, "coverage": None}, False),
    ("Дай мне {satelliteTextAcc}, которые видят {coverageAcc} и имеют статус {status}.", {"number": None, "coverage": None, "status": None}, False),
    ("Покажи мне {satelliteTextAcc} на {orbitType} с покрытием {coverageGen}.", {"number": None, "orbitType": None, "coverage": None}, False),
    ("Выведи, пожалуйста, {satelliteTextAcc} с массой {mass} и форм-фактором {formFactor}.", {"number": None, "mass": None, "formFactor": None}, False),
    ("Для наблюдения {coverageGen} требуется {satelliteTextNom} на {orbitType}.", {"number": None, "coverage": None, "orbitType": None}, False),
    ("Сформируй список {satelliteTextGen}, которые покрывают {coverageGen} и работают на {orbitType}.", {"number": None, "coverage": None, "orbitType": None}, False),
    ("Укажи {satelliteTextAcc} с массой не менее {mass} для {coverageGen}.", {"number": None, "mass": None, "coverage": None}, False),
    ("Ищу {satelliteTextAcc} для мониторинга {coverageGen} на высоте {altitude}.", {"number": None, "coverage": None, "altitude": None}, False),
    ("Покажи {satelliteTextAcc}, если масса равна {mass} и статус — {status}.", {"number": None, "mass": None, "status": None}, False),
    ("Если нужен {satelliteTextNom} с форм-фактором {formFactor}, что выбрать для {coverageGen}?", {"number": None, "formFactor": None, "coverage": None}, False),
    ("Найди {satelliteTextAcc}, обеспечивающих съемку {coverageGen} каждые 2 часа.", {"number": None, "coverage": None}, False),
    ("Для высоты {altitude} подбери {satelliteTextAcc}, которые видят {coverageAcc}.", {"number": None, "altitude": None, "coverage": None}, False),
    ("Покажи список {satelliteTextGen}, у которых статус {status}.", {"number": None, "status": None}, False),
    ("Сформируй запрос на {satelliteTextAcc} с покрытием {coverageGen} и высотой {altitude}.", {"number": None, "coverage": None, "altitude": None}, False),
    ("Сформируйте, пожалуйста, список {satelliteTextGen}, которые видят {coverageAcc}.", {"number": None, "coverage": None}, False),
    ("Выбери {satelliteTextAcc}, подходящие для мониторинга {coverageGen}.", {"number": None, "coverage": None}, False),
    ("Проверь наличие {satelliteTextGen} с форм-фактором {formFactor} и массой {mass}.", {"number": None, "formFactor": None, "mass": None}, False),
    ("Среди спутников с массой {mass}, какие покрывают {coverageGen}?", {"mass": None, "coverage": None}, False),
    ("Нужно подобрать {satelliteTextAcc}, которые работают над {coveragePrep}.", {"number": None, "coverage": None}, False),
    ("Среди {satelliteTextGen} выбери те, что имеют статус {status}.", {"number": None, "status": None}, False),
    ("Выведи {satelliteTextAcc} для задачи мониторинга {coverageGen} с высоты {altitude}.", {"number": None, "coverage": None, "altitude": None}, False),
    ("Список {satelliteTextGen} с TLE на дату {date} для {coverageGen}.", {"number": None, "date": None, "coverage": None}, False),
    ("Сформируй запрос на {satelliteTextAcc} с параметрами: масса {mass}, форм-фактор {formFactor}.", {"number": None, "mass": None, "formFactor": None}, False),
    ("Покажи {satelliteTextAcc}, способные к мониторингу {coverageGen} на {orbitType}.", {"number": None, "orbitType": None, "coverage": None}, False),
    ("Выведи, пожалуйста, {satelliteTextAcc} на {orbitType} для наблюдения {coverageGen} с массой {mass}.", {"number": None, "orbitType": None, "coverage": None, "mass": None}, False),
    ("Покажи {satelliteTextAcc}, которые могут быть использованы для съемки {coverageGen}.", {"number": None, "coverage": None}, False),
    ("Рассмотри {satelliteTextAcc} с форм-фактором {formFactor}, подходящие для {coverageGen}.", {"number": None, "formFactor": None, "coverage": None}, False),
    ("Определи {satelliteTextAcc}, которые работают на {orbitType} для {coverageGen}.", {"number": None, "orbitType": None, "coverage": None}, False),
    ("Список {satelliteTextGen} с массой {mass} и форм-фактором {formFactor} для {coverageGen}.", {"number": None, "mass": None, "formFactor": None, "coverage": None}, False),
    ("Покажи {satelliteTextAcc} для наблюдения {coverageGen} на дату {date}.", {"number": None, "coverage": None, "date": None}, False),
    ("Выбери {satelliteTextAcc}, которые функционируют над {coveragePrep}.", {"number": None, "coverage": None}, False),
    ("Для задачи наблюдения {coverageGen} предоставь {satelliteTextAcc} с форм-фактором {formFactor}.", {"number": None, "coverage": None, "formFactor": None}, False),
    ("Среди всех спутников выбери {satelliteTextAcc}, подходящие для {coverageGen}.", {"number": None, "coverage": None}, False),
    ("Выведи {satelliteTextAcc}, которые могут работать над {coveragePrep} на высоте {altitude}.", {"number": None, "coverage": None, "altitude": None}, False),
    ("Список {satelliteTextGen}, доступных для наблюдения {coverageGen} с форм-фактором {formFactor}.", {"number": None, "coverage": None, "formFactor": None}, False),
    ("Покажи {satelliteTextAcc} с массой не более {mass}, которые видят {coverageAcc}.", {"number": None, "mass": None, "coverage": None}, False),
    ("Для высоты {altitude} и покрытия {coverageGen} выбери {satelliteTextAcc}.", {"number": None, "altitude": None, "coverage": None}, False),
    ("Выведи {satelliteTextAcc} для мониторинга {coverageGen} на {orbitType} с массой {mass}.", {"number": None, "coverage": None, "orbitType": None, "mass": None}, False),
    ("Сформируй список {satelliteTextGen} для наблюдения {coverageGen} на дату {date}.", {"number": None, "coverage": None, "date": None}, False),

    # вопросы (True)
    ("Можешь подобрать {satelliteTextAcc} для наблюдения {coverageGen} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Какой {satelliteTextNom} доступен для наблюдения {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("С каким статусом есть {satelliteTextNom} над {coveragePrep}?", {"number": None, "coverage": None}, True),
    ("Какие {satelliteTextNom} доступны для покрытия {coverageGen} с TLE на {date}?", {"number": None, "coverage": None, "date": None}, True),
    ("Какие спутники с форм-фактором {formFactor} имеют массу {mass}?", {"formFactor": None, "mass": None}, True),
    ("Возможно ли подобрать {satelliteTextAcc} с форм-фактором {formFactor}?", {"number": None, "formFactor": None}, True),
    ("Какие параметры у {satelliteTextGen}, покрывающих {coverageGen}?", {"number": None, "coverage": None}, True),
    ("Какие {satelliteTextNom} могут быть использованы для наблюдения {coverageGen}?", {"number": None, "coverage": None}, True),
    ("Какие спутники на орбите {orbitType} работают над {coveragePrep}?", {"orbitType": None, "coverage": None}, True),
    ("Есть ли {satelliteTextNom} с массой {mass} для мониторинга {coverageGen}?", {"number": None, "mass": None, "coverage": None}, True),
    ("Какие {satelliteTextNom} соответствуют параметрам: масса {mass}, статус {status}?", {"number": None, "mass": None, "status": None}, True),
    ("На дату {date} какие доступны {satelliteTextNom} для наблюдения {coverageGen}?", {"number": None, "date": None, "coverage": None}, True),
    ("Можно ли получить {satelliteTextAcc} на {orbitType} с массой {mass}?", {"number": None, "orbitType": None, "mass": None}, True),
    ("Для задачи наблюдения {coverageGen} какие есть {satelliteTextNom} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Какие {satelliteTextNom} действуют над {coveragePrep} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Для покрытия {coverageGen} выбери {satelliteTextAcc} с форм-фактором {formFactor}.", {"number": None, "formFactor": None, "coverage": None}, True),
    ("Возможно ли использовать {satelliteTextAcc} для наблюдения {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("Если требуется масса {mass}, какие {satelliteTextNom} могут наблюдать {coverageGen}?", {"number": None, "mass": None, "coverage": None}, True),
    ("Какие спутники с форм-фактором {formFactor} и массой {mass} функционируют для {coverageGen}?", {"formFactor": None, "mass": None, "coverage": None}, True),
    ("Есть ли {satelliteTextNom} с покрытием {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("Какие есть {satelliteTextNom} для наблюдения {coverageGen} с массой {mass}?", {"number": None, "coverage": None, "mass": None}, True),
    ("Какие {satelliteTextNom} можно использовать для мониторинга {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("Какие {satelliteTextNom} с форм-фактором {formFactor} доступны для {coverageGen}?", {"number": None, "formFactor": None, "coverage": None}, True),
    ("Покажи спутники, подходящие для мониторинга {coverageGen} на высоте {altitude}.", {"coverage": None, "altitude": None}, True),
    ("Какие {satelliteTextNom} могут обеспечить наблюдение {coverageGen} каждые 2 часа?", {"number": None, "coverage": None}, True),
    ("Список {satelliteTextGen}, которые доступны для наблюдения {coverageGen}.", {"number": None, "coverage": None}, True),
    ("Можно ли подобрать {satelliteTextAcc} для покрытия {coverageGen} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Какие спутники на {orbitType} могут вести мониторинг {coverageGen} с массой {mass}?", {"orbitType": None, "coverage": None, "mass": None}, True),
    ("Какие {satelliteTextNom} могут захватывать изображения {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("Какие есть {satelliteTextNom} с массой {mass} на {orbitType}?", {"number": None, "mass": None, "orbitType": None}, True),
    ("Какие спутники способны обеспечить покрытие {coverageGen} с высоты {altitude}?", {"coverage": None, "altitude": None}, True),
    ("Какие из {satelliteTextGen} работают над {coveragePrep} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Можно ли подключить {satelliteTextAcc} для мониторинга {coverageGen}?", {"number": None, "coverage": None}, True),
    ("Для высоты {altitude}, покрытия {coverageGen} и массы {mass} покажи {satelliteTextAcc}.", {"number": None, "altitude": None, "coverage": None, "mass": None}, True),
    ("С каким TLE доступны {satelliteTextNom} для {coverageGen} на дату {date}?", {"number": None, "coverage": None, "date": None}, True),
    ("Какие {satelliteTextNom} можно использовать для съемки {coverageGen} на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Можно ли получить список {satelliteTextGen} для мониторинга {coverageGen} на высоте {altitude}?", {"number": None, "coverage": None, "altitude": None}, True),
    ("Какой {satelliteTextNom} с массой {mass} подходит для наблюдения {coverageGen}?", {"number": None, "mass": None, "coverage": None}, True),
    ("Какие {satelliteTextNom} имеют статус {status} и покрывают {coverageGen}?", {"number": None, "status": None, "coverage": None}, True),
    ("Для задачи мониторинга {coverageGen} укажи {satelliteTextAcc} на {orbitType}.", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Можно ли подобрать {satelliteTextAcc} для {coverageGen}, работающие на {orbitType}?", {"number": None, "coverage": None, "orbitType": None}, True),
    ("Какие {satelliteTextNom} доступны для съемки {coverageGen} на {orbitType} с форм-фактором {formFactor}?", {"number": None, "orbitType": None, "coverage": None, "formFactor": None}, True),
    ("Покажи {satelliteTextAcc}, если масса {mass} и статус {status}.", {"number": None, "mass": None, "status": None}, True),
    ("Какие {satelliteTextNom} соответствуют форм-фактору {formFactor} и массе {mass}?", {"number": None, "formFactor": None, "mass": None}, True),
    ("Какие есть {satelliteTextNom} с массой {mass} для мониторинга {coverageGen}?", {"number": None, "mass": None, "coverage": None}, True),
    ("Какие спутники с форм-фактором {formFactor} могут работать на {orbitType} для {coverageGen}?", {"formFactor": None, "orbitType": None, "coverage": None}, True),
    ("Среди {satelliteTextGen} выбери те, что видят {coverageAcc} на высоте {altitude}.", {"number": None, "coverage": None, "altitude": None}, True),
    ("Для наблюдения {coverageGen} выдай {satelliteTextAcc} на высоте {altitude}.", {"number": None, "coverage": None, "altitude": None}, True),
]

def random_element(seq):
    return random.choice(seq)

def random_element_key(dct):
    return random.choice(list(dct.keys()))

def random_date():
    days_shift = random.randint(-365, 365)
    return (datetime.now() + timedelta(days=days_shift)).strftime("%Y-%m-%d")

def determine_scale(mass):
    return "малый" if mass <= 100 else "большой"

def random_satellite_number():
    # 90% — один спутник, 10% — от 2 до 20
    if random.random() < 0.90:
        return 1
    else:
        return random.randint(2, 20)

def satellite_text(number, case_type, as_word):
    if as_word and number <= 20:
        num_str = {
            "nom": numbers_words_nom, "acc": numbers_words_acc,
            "gen": numbers_words_gen, "dat": numbers_words_dat, "prep": numbers_words_prep
        }[case_type].get(number, str(number))
    else:
        num_str = str(number)
    last_digit = number % 10
    last_two = number % 100
    if 11 <= last_two <= 19:
        word = {
            "nom": "спутников", "acc": "спутников", "gen": "спутников", "dat": "спутникам", "prep": "спутникам"
        }[case_type]
    else:
        if last_digit == 1:
            word = {
                "nom": "спутник", "acc": "спутник", "gen": "спутника", "dat": "спутнику", "prep": "спутнике"
            }[case_type]
        elif last_digit in (2, 3, 4):
            word = {
                "nom": "спутника", "acc": "спутника", "gen": "спутников", "dat": "спутникам", "prep": "спутникам"
            }[case_type]
        else:
            word = {
                "nom": "спутников", "acc": "спутников", "gen": "спутников", "dat": "спутникам", "prep": "спутникам"
            }[case_type]
    return f"{num_str} {word}"

def is_valid_combination(orbit_type, coverage):
    if orbit_type == "GEO" and coverage == "Арктика":
        return False
    if orbit_type == "Molniya" and coverage in ["Африка", "Южная Америка"]:
        return False
    if orbit_type == "HEO" and coverage in ["Африка", "Южная Америка"]:
        return False
    return True

def is_basic_grammar_valid(text):
    if "  " in text:
        return False
    if not any(p in text for p in "?.!,"):
        return False
    if len(text.strip()) < 10:
        return False
    if any("a" <= c <= "z" or "A" <= c <= "Z" for c in text):
        return False
    verbs = ["Найди", "Подбери", "Покажи", "Ищи", "Выведи", "Скинь", "Какие", "Нужны", "Требуется", "Дай", "Обеспечь", "Предложи", "Предоставь", "Подскажи"]
    return text.startswith(tuple(verbs)) or any(v.lower() in text.lower() for v in verbs)

def is_physically_valid(filters):
    ff = filters.get("formFactor")
    mass = filters.get("mass")
    if ff and mass:
        try:
            val = int(mass.split()[0])
            minv, maxv = valid_masses_for_form_factor[ff]
            return minv <= val <= maxv
        except Exception:
            return False
    return True

def yandex_speller_check(text):
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {
        "text": text,
        "lang": "ru",
    }
    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        result = response.json()
        return len(result) == 0
    except Exception as e:
        print(f"Spellcheck error: {e}")
        return True  # В случае ошибки API считаем промт валидным

def morph_check_case(text):
    errors = []
    for prep in ["на", "над", "в", "под"]:
        pat = re.compile(rf"{prep} ([^.,\n]+?(?:орбита|орбиту|орбите|региону|регионе|регион|полюсе|полюс|Китае|Китаю|Китай|России|Европе|Африке))", re.IGNORECASE)
        for m in pat.finditer(text):
            phrase = m.group(1).strip().split()
            if phrase:
                last_word = phrase[-1]
                parsed = morph.parse(last_word)[0]
                if prep in ["на", "в"]:
                    if parsed.tag.case != 'loct':
                        errors.append(f"'{last_word}' не в предложном падеже после '{prep}'")
                if prep in ["над", "под"]:
                    if parsed.tag.case != 'ablt':
                        errors.append(f"'{last_word}' не в творительном падеже после '{prep}'")
    pat = re.compile(r"(над|под)\s+([А-Яа-яЁё]+)", re.IGNORECASE)
    for m in pat.finditer(text):
        prep, word = m.group(1), m.group(2)
        parsed = morph.parse(word)[0]
        if 'NOUN' in parsed.tag and 'Infr' not in parsed.tag:
            if parsed.tag.case != 'ablt':
                errors.append(f"'{word}' не в творительном падеже после '{prep}'")
    return errors

def morph_check_number_agreement(text):
    errors = []
    if re.search(r'1 спутник[а-я]*[оы]в', text):
        errors.append("Несогласование: '1 спутников' — должно быть '1 спутник'")
    if re.search(r'1 спутник.*которые', text):
        errors.append("Несогласование: '1 спутник' — должно быть 'который'")
    if re.search(r'список 1 спутников', text):
        errors.append("Несогласование: 'список 1 спутников' — должно быть 'список одного спутника'")
    if re.search(r'один спутник.*(которые|доступные|работающие|имеют)', text):
        errors.append("Несогласование: 'один спутник' — глагол/местоимение должно быть в единственном числе")
    if re.search(r'Какие.*один спутник', text, re.IGNORECASE):
        errors.append("Несогласование: 'Какие один спутник' — должно быть 'Какой спутник'")
    # Если "который могут" или "который могут обеспечить" после единственного числа
    if re.search(r'(?:1|один|об одном|одному|одного|одним|одном) спутник[^\n.,]*котор(ый|ое|ая|ое|ые)[^\n.,]*могут', text):
        errors.append("Несогласование: 'который могут' — должно быть 'который может' для единственного числа")
    return errors

def is_prompt_high_quality(prompt):
    if has_repeated_words(prompt):
        return False
    if has_double_units(prompt):
        return False
    if has_number_agreement_issue(prompt):
        return False
    if has_redundant_phrases(prompt):
        return False
    if has_bad_mass_phrasing(prompt):
        return False
    if morph_check_case(prompt):
        return False
    if morph_check_number_agreement(prompt):
        return False
    return True

def has_repeated_words(text):
    words = text.lower().split()
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            return True
    return False

def has_double_units(text):
    return bool(re.search(r'\b(кг|км|шт|градусов|спутник|высота|масса|орбита)(\s+\1)+\b', text))

def has_number_agreement_issue(text):
    return bool(re.search(r'какие\s+1\s+спутник', text, re.IGNORECASE))

def has_redundant_phrases(text):
    return bool(re.search(r'\b([а-яё]+)\s+\1\b', text, re.IGNORECASE))

def has_bad_mass_phrasing(text):
    return bool(re.search(r'кг\s+кг', text))

def fix_status_word(status, number):
    singular_map = {
        "активен": "активный",
        "активный": "активный",
        "работающий": "работающий",
        "функционирующий": "функционирующий",
        "неактивен": "неактивный",
        "неактивный": "неактивный",
        "неработающий": "неработающий",
        "нефункционирующий": "нефункционирующий",
        "вышедший из строя": "вышедший из строя",
    }
    plural_map = {
        "активен": "активные",
        "активный": "активные",
        "работающий": "работающие",
        "функционирующий": "функционирующие",
        "неактивен": "неактивные",
        "неактивный": "неактивные",
        "неработающий": "неработающие",
        "нефункционирующий": "нефункционирующие",
        "вышедший из строя": "вышедшие из строя",
    }
    status = status.strip()
    if int(number) == 1:
        return singular_map.get(status, status)
    else:
        return plural_map.get(status, status)

def fix_list_phrase(prompt, number):
    if re.search(r"список 1 спутника", prompt):
        if int(number) == 1:
            prompt = re.sub(r"список 1 спутника", "информацию об одном спутнике", prompt)
        else:
            prompt = re.sub(r"список 1 спутника", "список спутников", prompt)
    if re.search(r"список 2 спутников", prompt):
        prompt = re.sub(r"список 2 спутников", "список двух спутников", prompt)
    # Исправляем "Нужны информацию" на "Нужна информация"
    prompt = re.sub(r"Нужны информацию", "Нужна информация", prompt)
    return prompt

def fix_kakie_odin(prompt, number):
    if int(number) == 1:
        prompt = re.sub(r"Какие один спутник", "Какой спутник", prompt, flags=re.IGNORECASE)
    return prompt

def fix_status_plural(prompt, number):
    def repl(match):
        word = match.group(1)
        return "статус " + fix_status_word(word, number)
    prompt = re.sub(r"статус( [а-яА-ЯёЁ ]+?)([,.])", lambda m: repl(m) + m.group(2), prompt)
    return prompt

def fix_kotorye_kotoryj(prompt, number):
    if int(number) == 1:
        prompt = re.sub(r"которые", "который", prompt)
        prompt = re.sub(r"работают", "работает", prompt)
        prompt = re.sub(r"имеют", "имеет", prompt)
        prompt = re.sub(r"доступные", "доступен", prompt)
        # Исправляем "который могут" на "который может"
        prompt = re.sub(r"который могут", "который может", prompt)
        prompt = re.sub(r"которое могут", "которое может", prompt)
    return prompt

def fix_russian_prompt(prompt, filters):
    # 1. Повтор "высота высота"
    prompt = re.sub(r'высоте\s+высота', 'высоте', prompt)
    prompt = re.sub(r'на высоте высота', 'на высоте', prompt)
    prompt = re.sub(r'высоте высота', 'высоте', prompt)
    # 2. "Какие есть 1 спутник"
    prompt = re.sub(r'Какие есть 1 спутник', 'Какой спутник', prompt)
    prompt = re.sub(r'Какие есть один спутник', 'Какой спутник', prompt)
    prompt = re.sub(r'Какие есть (\d+) спутник', r'Какие спутники', prompt) # если >1
    # 3. "Требуется информацию"
    prompt = re.sub(r'Требуется информацию', 'Требуется информация', prompt)
    # 4. "выдай 1 спутник"
    prompt = re.sub(r'выдай 1 спутник', 'выдай одного спутника', prompt)
    prompt = re.sub(r'выдай один спутник', 'выдай спутник', prompt)
    # 5. "покажи 1 спутник"
    prompt = re.sub(r'покажи 1 спутник', 'покажи одного спутника', prompt)
    prompt = re.sub(r'покажи один спутник', 'покажи спутник', prompt)
    # 6. "предложи 1 спутник"
    prompt = re.sub(r'предложи 1 спутник', 'предложи одного спутника', prompt)
    prompt = re.sub(r'предложи один спутник', 'предложи спутник', prompt)
    # 7. "Скинь мне спутники" без числа — корректно, но если число 1, лучше "спутник"
    if filters.get('number') == "1":
        prompt = re.sub(r'спутники, подходящие', 'спутник, подходящий', prompt)
        prompt = re.sub(r'спутники для', 'спутник для', prompt)
        prompt = re.sub(r'подходящие для', 'подходящий для', prompt)
    # 8. "покажи 1 спутник"
    prompt = re.sub(r'покажи 1 спутник', 'покажи одного спутника', prompt)
    # 9. "выдай 1 спутник"
    prompt = re.sub(r'выдай 1 спутник', 'выдай одного спутника', prompt)
    # 10. "Скинь мне спутники" если число 1
    if filters.get('number') == "1":
        prompt = re.sub(r'Скинь мне спутники', 'Скинь мне спутник', prompt)
    # 11. Избавиться от двойных пробелов
    prompt = re.sub(r' +', ' ', prompt)
    # 12. Корректировка "с массой 10 кг" vs "масса 10 кг"
    prompt = re.sub(r'с массой ([\d]+ кг)', r'масса \1', prompt)
    # 13. Начало с "Какие есть" и 1 спутник
    if re.match(r'Какие есть [1-9] спутник', prompt) or re.match(r'Какие есть один спутник', prompt):
        prompt = re.sub(r'Какие есть [1-9] спутник', 'Какой спутник', prompt)
        prompt = re.sub(r'Какие есть один спутник', 'Какой спутник', prompt)
    prompt = prompt.strip()
    return prompt

def postprocess_prompt(prompt, filters):
    number = filters.get("number", "1")
    prompt = fix_russian_prompt(prompt, filters)
    prompt = fix_list_phrase(prompt, number)
    prompt = fix_kakie_odin(prompt, number)
    prompt = fix_status_plural(prompt, number)
    prompt = fix_kotorye_kotoryj(prompt, number)
    return prompt

def is_prompt_reasonable(prompt, filters):
    prompt = postprocess_prompt(prompt, filters)
    return (
        is_basic_grammar_valid(prompt) and
        yandex_speller_check(prompt) and
        is_physically_valid(filters) and
        is_prompt_high_quality(prompt)
    )

def generate_prompt():
    # 70% — утверждение, 30% — вопрос
    want_question = random.random() > 0.7
    # выбираем шаблоны нужного типа
    candidates = [tpl for tpl in templates if tpl[2] == want_question]
    template, required_filters, _ = random.choice(candidates)
    selected_filters = {}
    text_values = {}

    while True:
        orbit_type = required_filters.get("orbitType") or random_element_key(orbit_types)
        coverage = required_filters.get("coverage") or random_element_key(coverages)
        if is_valid_combination(orbit_type, coverage):
            if required_filters.get("orbitType") is None:
                selected_filters["orbitType"] = orbit_type
            if required_filters.get("coverage") is None:
                selected_filters["coverage"] = coverage
            break

    if "altitude" in required_filters:
        orbit_for_alt = selected_filters.get("orbitType") or random_element_key(valid_altitudes_for_orbit)
        selected_filters["altitude"] = random.choice(valid_altitudes_for_orbit[orbit_for_alt])

    if "formFactor" in required_filters:
        selected_filters["formFactor"] = random_element_key(form_factors)

    if "mass" in required_filters:
        ff_for_mass = selected_filters.get("formFactor") or random_element_key(valid_masses_for_form_factor)
        minv, maxv = valid_masses_for_form_factor[ff_for_mass]
        selected_filters["mass"] = str(random.randint(minv, maxv))

    if "status" in required_filters:
        val = required_filters["status"]
        selected_filters["status"] = val if val is not None else random_element_key(statuses)

    if "date" in required_filters:
        selected_filters["date"] = random_date()

    if "scale" in required_filters:
        val = required_filters["scale"]
        if val:
            selected_filters["scale"] = val
        else:
            massval = int(selected_filters.get("mass", 10))
            selected_filters["scale"] = determine_scale(massval)

    if "number" in required_filters or "{satelliteText" in template:
        number = int(required_filters.get("number") or random_satellite_number())
        selected_filters["number"] = str(number)
        use_word = random.random() < 0.20 and number <= 20
        text_values["number"] = str(number)
        text_values["satelliteTextNom"] = satellite_text(number, "nom", use_word)
        text_values["satelliteTextAcc"] = satellite_text(number, "acc", use_word)
        text_values["satelliteTextGen"] = satellite_text(number, "gen", use_word)
        text_values["satelliteTextDat"] = satellite_text(number, "dat", use_word)
        text_values["satelliteTextPrep"] = satellite_text(number, "prep", use_word)

    for k, v in selected_filters.items():
        if k == "coverage":
            cm = coverages[v]
            text_values["coverageNom"] = random.choice(cm["nom"])
            text_values["coverageGen"] = random.choice(cm["gen"])
            text_values["coverageAcc"] = random.choice(cm["acc"])
            text_values["coverageDat"] = random.choice(cm["dat"])
            text_values["coveragePrep"] = random.choice(cm["prep"])
        elif k == "altitude":
            text_values["altitude"] = random.choice(altitudes[v])
        elif k == "orbitType":
            text_values["orbitType"] = random.choice(orbit_types[v])
        elif k == "status":
            text_values["status"] = fix_status_word(random.choice(statuses[v]), selected_filters.get("number", "1"))
        elif k == "formFactor":
            text_values["formFactor"] = random.choice(form_factors[v])
        elif k == "mass":
            text_values["mass"] = f"{v} кг"
        elif k == "date":
            text_values["date"] = v
        elif k == "scale":
            text_values["scale"] = v

    filled_text = template
    for k, v in text_values.items():
        filled_text = filled_text.replace(f"{{{k}}}", v)

    first_word = filled_text.split(' ')[0]
    if first_word.lower() in [s.lower() for s in request_synonyms]:
        new_start = random.choice(request_synonyms)
        if random.random() < 0.3 and new_start not in ["Какие есть", "Требуется", "Нужны"]:
            new_start += " мне"
        filled_text = filled_text.replace(first_word, new_start, 1)

    filled_text = postprocess_prompt(filled_text, selected_filters)
    return filled_text, selected_filters

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    count = args.count
    seen_prompts = set()
    prompts = []
    attempts = 0
    rejected = 0
    max_attempts = count * 30

    while len(prompts) < count and attempts < max_attempts:
        attempts += 1
        prompt, filters = generate_prompt()
        prompt_key = (prompt + str(sorted(filters.items()))).lower().replace("  ", " ").strip()
        if prompt_key and prompt_key not in seen_prompts and is_prompt_reasonable(prompt, filters):
            seen_prompts.add(prompt_key)
            prompts.append({"prompt": prompt, "filters": filters})
        else:
            rejected += 1
            
        # Информирование пользователя о прогрессе каждые 10 успешных промтов
        if len(prompts) % 10 == 0 or len(prompts) == count:
            percent = 100 * len(prompts) // count
            sys.stdout.write(f"\rСгенерировано {len(prompts)} из {count} ({percent}%) промтов, попыток: {attempts}, отброшено: {rejected}")
            sys.stdout.flush()

    with open("prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Сгенерировано {len(prompts)} уникальных промптов в файл 'prompts.jsonl'")
    print(f"Отброшено невалидных промптов: {rejected}")
    print(f"Всего попыток генерации: {attempts}")

if __name__ == "__main__":
    main()