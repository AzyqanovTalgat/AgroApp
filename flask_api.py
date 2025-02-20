import os
import traceback
import random

import joblib
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================================
# 1) НАСТРОЙКА ПУТЕЙ И ЗАГРУЗКА МОДЕЛЕЙ
# =========================================

# Определяем базовую директорию, где лежит текущий файл
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути к моделям
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "simple_classification_model.pkl")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "shap_weighted_models.pkl")

# Загрузка моделей
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
regression_models = joblib.load(REGRESSION_MODEL_PATH)

# Пути к маппингам (текстовые файлы)
CROP_MAPPING_PATH = os.path.join(BASE_DIR, "crop_mapping.txt")
SOIL_MAPPING_PATH = os.path.join(BASE_DIR, "soil_mapping.txt")

# Функция для загрузки маппингов
def load_mapping(filepath):
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            idx, name = line.strip().split(" ", 1)
            mapping[int(idx)] = name
    return mapping

# Загрузка маппингов
crop_mapping = load_mapping(CROP_MAPPING_PATH)
soil_mapping = load_mapping(SOIL_MAPPING_PATH)

# =========================================
# 2) НАСТРОЙКА ПРИЛОЖЕНИЯ FLASK
# =========================================

app = Flask(__name__)
CORS(app)

# =========================================
# 3) СПИСКИ ПРИЗНАКОВ ДЛЯ МОДЕЛЕЙ
# =========================================
FEATURE_NAMES_CLASS = [
    "soil_type","prev_crop_1","prev_crop_2","prev_crop_3",
    "area_ha","soil_ph","temp_avg","rainfall_mm",
    "fert_cost","fuel_cost","seed_cost","market_price",
    "transp_cost","num_workers",
    "humus_percent","nitrogen_mg_kg","phosphorus_mg_kg","potassium_mg_kg",
    "soil_moisture","soil_density","permeability","fertile_depth",
    "salinity","stoniness","erosion",
]

FEATURE_NAMES_REG = [
    "soil_type","rec_crop","prev_crop_1","prev_crop_2","prev_crop_3",
    "area_ha","soil_ph","temp_avg","rainfall_mm",
    "fert_cost","fuel_cost","seed_cost","market_price",
    "transp_cost","num_workers",
    "humus_percent","nitrogen_mg_kg","phosphorus_mg_kg","potassium_mg_kg",
    "soil_moisture","soil_density","permeability","fertile_depth",
    "salinity","stoniness","erosion",
]

# =========================================
# 4) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПАРСИНГА
# =========================================
def parse_features_for_class(req):
    """Считывает данные из формы (req.form) и приводит к нужному формату для классификационной модели."""
    feats = {}
    for n in FEATURE_NAMES_CLASS:
        v = req.form.get(n, "0")
        try:
            feats[n] = float(v)
        except:
            feats[n] = 0.0
    return pd.DataFrame([feats])[FEATURE_NAMES_CLASS]

def parse_features_for_reg(req):
    """Считывает данные из формы (req.form) и приводит к нужному формату для регрессионных моделей."""
    feats = {}
    for n in FEATURE_NAMES_REG:
        v = req.form.get(n, "0")
        try:
            feats[n] = float(v)
        except:
            feats[n] = 0.0
    return pd.DataFrame([feats])[FEATURE_NAMES_REG]

# =========================================
# 5) РОУТЫ: /api/predict_class /api/predict_reg /api/full_prediction
# =========================================

@app.route("/api/predict_class", methods=["POST"])
def predict_class():
    """Предсказание класса (культуры) + вероятности."""
    try:
        X_class = parse_features_for_class(request)
        y_pred = classification_model.predict(X_class)[0]
        y_proba = classification_model.predict_proba(X_class)[0].tolist()

        crop_name = crop_mapping.get(int(y_pred), "Неизвестно")

        return jsonify({
            "predicted_class": int(y_pred),
            "predicted_crop_name": crop_name,
            "probabilities": y_proba
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_reg", methods=["POST"])
def predict_reg():
    """Предсказание нескольких регрессионных таргетов сразу."""
    try:
        X_reg = parse_features_for_reg(request)
        out = {}
        for tgt, mdl in regression_models.items():
            out[tgt] = float(mdl.predict(X_reg)[0])
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/full_prediction", methods=["POST"])
def full_prediction():
    """
    Сначала классификационная модель определяет rec_crop,
    затем этот rec_crop подставляется в регрессионную модель.
    Возвращаем все таргеты + рекомендуемую культуру.
    """
    try:
        # 1. Предсказание культуры
        X_class = parse_features_for_class(request)
        rec_crop = classification_model.predict(X_class)[0]
        rec_crop_val = int(rec_crop)
        rec_crop_name = crop_mapping.get(rec_crop_val, "???")

        # 2. Подготовка к регрессии: rec_crop добавляем в признаки
        X_reg = parse_features_for_reg(request)
        X_reg["rec_crop"] = rec_crop_val

        # 3. Прогоняем все регрессионные модели
        all_preds = {}
        for tgt, mdl in regression_models.items():
            all_preds[tgt] = float(mdl.predict(X_reg)[0])
        all_preds["recommended_crop"] = rec_crop_name

        return jsonify(all_preds)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================================
# 6) РОУТ /api/get_scenarios (демо-генерация сценариев)
# =========================================
@app.route("/api/get_scenarios", methods=["POST"])
def get_scenarios():
    """
    Пример генерации случайных сценариев, расчёта score и выбора топ-3.
    В POST-запросе ожидаем JSON:
    {
        "chosen_targets": [...],   # какие метрики суммируем в score
        "soil_type": int или -1,
        "prev_crop_1": int или -1,
        "prev_crop_2": int или -1,
        "prev_crop_3": int или -1,
        "area_ha": int или -1,
        "distinct_crops": bool
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error":"No JSON data provided"}), 400

        chosen_targets = data.get("chosen_targets", [])
        soil_tp = data.get("soil_type", -1)
        p1 = data.get("prev_crop_1", -1)
        p2 = data.get("prev_crop_2", -1)
        p3 = data.get("prev_crop_3", -1)
        area_val = data.get("area_ha", -1)
        distinct_crops = data.get("distinct_crops", False)

        N = 50
        scenarios = []
        for _ in range(N):
            # Формируем признаки для классификации
            fclass = {}
            # soil_type
            if soil_tp != -1:
                fclass["soil_type"] = float(soil_tp)
            else:
                fclass["soil_type"] = float(random.randint(0, 5))

            # prev_crop_i
            if p1 != -1:
                fclass["prev_crop_1"] = float(p1)
            else:
                fclass["prev_crop_1"] = float(random.randint(0, 6))

            if p2 != -1:
                fclass["prev_crop_2"] = float(p2)
            else:
                fclass["prev_crop_2"] = float(random.randint(0, 6))

            if p3 != -1:
                fclass["prev_crop_3"] = float(p3)
            else:
                fclass["prev_crop_3"] = float(random.randint(0, 6))

            # area_ha
            if area_val > 0:
                low = max(1, area_val - 1)
                high = min(50000, area_val + 1)
                a_ha = random.randint(low, high)
                fclass["area_ha"] = float(a_ha)
            else:
                a_ha = random.randint(10, 50000)
                fclass["area_ha"] = float(a_ha)

            # Остальные признаки для классификации
            fclass["soil_ph"]       = round(random.uniform(5, 7.5), 2)
            fclass["temp_avg"]      = round(random.uniform(5, 25), 1)
            fclass["rainfall_mm"]   = float(random.randint(100, 800))
            fclass["fert_cost"]     = float(random.randint(100, 500))
            fclass["fuel_cost"]     = float(random.randint(150, 600))
            fclass["seed_cost"]     = float(random.randint(50, 250))
            fclass["market_price"]  = float(random.randint(8000,40000))
            fclass["transp_cost"]   = float(random.randint(50, 1000))
            fclass["num_workers"]   = float(random.randint(1, 50))

            fclass["humus_percent"]    = round(random.uniform(2,10), 2)
            fclass["nitrogen_mg_kg"]   = round(random.uniform(50,200),2)
            fclass["phosphorus_mg_kg"] = round(random.uniform(10,50),2)
            fclass["potassium_mg_kg"]  = round(random.uniform(100,400),2)
            fclass["soil_moisture"]    = round(random.uniform(10,40),2)
            fclass["soil_density"]     = round(random.uniform(0.8,1.6),2)
            fclass["permeability"]     = round(random.uniform(2,15),2)
            fclass["fertile_depth"]    = round(random.uniform(15,40),2)
            fclass["salinity"]         = float(random.randint(1,3))
            fclass["stoniness"]        = float(random.randint(1,3))
            fclass["erosion"]          = float(random.randint(1,3))

            # Классификация -> rec_crop
            df_class = pd.DataFrame([fclass])[FEATURE_NAMES_CLASS]
            rec_crop = classification_model.predict(df_class)[0]
            rec_crop_val = int(rec_crop)

            # Подготовка данных для регрессии
            freg = {
                "soil_type":     fclass["soil_type"],
                "rec_crop":      float(rec_crop_val),
                "prev_crop_1":   fclass["prev_crop_1"],
                "prev_crop_2":   fclass["prev_crop_2"],
                "prev_crop_3":   fclass["prev_crop_3"],
                "area_ha":       fclass["area_ha"],
                "soil_ph":       fclass["soil_ph"],
                "temp_avg":      fclass["temp_avg"],
                "rainfall_mm":   fclass["rainfall_mm"],
                "fert_cost":     fclass["fert_cost"],
                "fuel_cost":     fclass["fuel_cost"],
                "seed_cost":     fclass["seed_cost"],
                "market_price":  fclass["market_price"],
                "transp_cost":   fclass["transp_cost"],
                "num_workers":   fclass["num_workers"],
                "humus_percent": fclass["humus_percent"],
                "nitrogen_mg_kg":   fclass["nitrogen_mg_kg"],
                "phosphorus_mg_kg": fclass["phosphorus_mg_kg"],
                "potassium_mg_kg":  fclass["potassium_mg_kg"],
                "soil_moisture":    fclass["soil_moisture"],
                "soil_density":     fclass["soil_density"],
                "permeability":     fclass["permeability"],
                "fertile_depth":    fclass["fertile_depth"],
                "salinity":         fclass["salinity"],
                "stoniness":        fclass["stoniness"],
                "erosion":          fclass["erosion"],
            }

            df_reg = pd.DataFrame([freg])[FEATURE_NAMES_REG]

            # Прогон по регрессионным моделям
            preds = {}
            for tgt, mdl in regression_models.items():
                preds[tgt] = float(mdl.predict(df_reg)[0])

            # score по выбранным целям
            score_val = 0.0
            for ct in chosen_targets:
                if ct in preds:
                    score_val += preds[ct]

            scenario_dict = {
                **fclass,
                "rec_crop": rec_crop_val,
                **preds,
                "score": round(score_val,4)
            }
            scenarios.append(scenario_dict)

        # Сортируем по score убыванию
        scenarios_sorted = sorted(scenarios, key=lambda x: x["score"], reverse=True)

        # Выбор top-3
        if distinct_crops:
            # Берём максимум один сценарий на каждую культуру
            best_by_crop = {}
            for scn in scenarios_sorted:
                rc = scn["rec_crop"]
                if rc not in best_by_crop:
                    best_by_crop[rc] = scn
            unique_scenarios = list(best_by_crop.values())
            unique_scenarios_sorted = sorted(unique_scenarios, key=lambda x: x["score"], reverse=True)
            topN = unique_scenarios_sorted[:3]
        else:
            # Просто берём первые три
            topN = scenarios_sorted[:3]

        # Формируем финальный ответ
        final_results = []
        for scn in topN:
            scn_copy = dict(scn)

            # Удаляем prev_crop_1..3, если не нужны
            for pc in ["prev_crop_1","prev_crop_2","prev_crop_3"]:
                if pc in scn_copy:
                    del scn_copy[pc]

            # Удаляем уже рассчитанные chosen_targets из вывода
            for ct in chosen_targets:
                if ct in scn_copy:
                    del scn_copy[ct]

            # Преобразуем soil_type и rec_crop в названия
            if "soil_type" in scn_copy:
                st_code = int(scn_copy["soil_type"])
                scn_copy["soil_type"] = soil_mapping.get(st_code, "???")

            if "rec_crop" in scn_copy:
                rc = int(scn_copy["rec_crop"])
                scn_copy["rec_crop"] = crop_mapping.get(rc, "???")

            final_results.append(scn_copy)

        return jsonify({"top_scenarios": final_results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================================
# 7) ТЕСТОВЫЙ РОУТ (главная страница)
# =========================================
@app.route("/", methods=["GET"])
def index():
    """Просто проверка, что сервер работает."""
    return jsonify({"message": "Dynamic Flask API with distinct crops check is running OK."}), 200

# =========================================
# 8) ЗАПУСК ПРИЛОЖЕНИЯ (локально)
# =========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
