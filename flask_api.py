# =========================== flask_app.py ===========================
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import pandas as pd
import numpy as np
import random
import os
app = Flask(__name__)
CORS(app)

# ====================== 1) ЗАГРУЗКА МОДЕЛЕЙ ======================
classification_model = joblib.load("/Users/talgatazykanov/Desktop/Science2023/Asemgul_PhD/Python_app/models/ensemble_classification_model.pkl")
regression_models    = joblib.load("/Users/talgatazykanov/Desktop/Science2023/Asemgul_PhD/Python_app/models/catb_ensemble_shap_weighted_models.pkl")

# ====================== 2) ЗАГРУЗКА МАППИНГОВ ======================
def load_mapping(filepath):
    mp={}
    with open(filepath,"r",encoding="utf-8") as f:
        for line in f:
            idx,name=line.strip().split(" ",1)
            mp[int(idx)]=name
    return mp

crop_mapping = load_mapping("crop_mapping.txt")  
soil_mapping = load_mapping("soil_mapping.txt")

# ====================== 3) СПИСКИ ПРИЗНАКОВ (CLASS/REG) ======================
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

# ====================== 4) /api/predict_class, /api/predict_reg, /api/full_prediction ======================
def parse_features_for_class(req):
    feats={}
    for n in FEATURE_NAMES_CLASS:
        v=req.form.get(n,"0")
        try:
            feats[n]=float(v)
        except:
            feats[n]=0.0
    return pd.DataFrame([feats])[FEATURE_NAMES_CLASS]

def parse_features_for_reg(req):
    feats={}
    for n in FEATURE_NAMES_REG:
        v=req.form.get(n,"0")
        try:
            feats[n]=float(v)
        except:
            feats[n]=0.0
    return pd.DataFrame([feats])[FEATURE_NAMES_REG]

@app.route("/api/predict_class", methods=["POST"])
def predict_class():
    try:
        X_class=parse_features_for_class(request)
        y_pred=classification_model.predict(X_class)[0]
        y_proba=classification_model.predict_proba(X_class)[0].tolist()
        crop_name=crop_mapping.get(int(y_pred),"Неизвестно")

        return jsonify({
            "predicted_class":int(y_pred),
            "predicted_crop_name":crop_name,
            "probabilities":y_proba
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/api/predict_reg", methods=["POST"])
def predict_reg():
    try:
        X_reg=parse_features_for_reg(request)
        out={}
        for tgt,mdl in regression_models.items():
            out[tgt]=float(mdl.predict(X_reg)[0])
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/api/full_prediction", methods=["POST"])
def full_prediction():
    try:
        X_class=parse_features_for_class(request)
        rec_crop=classification_model.predict(X_class)[0]
        rec_crop_val=int(rec_crop)
        rec_crop_name=crop_mapping.get(rec_crop_val,"???")

        X_reg=parse_features_for_reg(request)
        X_reg["rec_crop"]=rec_crop_val

        all_preds={}
        for tgt,mdl in regression_models.items():
            all_preds[tgt]=float(mdl.predict(X_reg)[0])
        all_preds["recommended_crop"]=rec_crop_name

        return jsonify(all_preds)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

# ====================== 5) /api/get_scenarios ======================
@app.route("/api/get_scenarios", methods=["POST"])
def get_scenarios():
    """
    Принимаем JSON:
    {
      "chosen_targets": [...],
      "soil_type": int (или -1),
      "prev_crop_1": int (или -1),
      "prev_crop_2": int (или -1),
      "prev_crop_3": int (или -1),
      "area_ha": int (или -1),
      "distinct_crops": bool (true/false)
    }
    Генерируем N=2000 сценариев:
      - Учитываем soil_type, prev_crop_i, area_ha±100
      - Сначала classification_model -> rec_crop
      - Потом regression_models -> preds
      - score = сумма выбранных target'ов
    Если distinct_crops=true => берём из отсортированного списка 
      только по 1 сценарию на каждую культуру => потом top-3
    Иначе => top-3 в общем виде.
    """
    try:
        data=request.get_json()
        if not data:
            return jsonify({"error":"No JSON data provided"}),400

        chosen_targets=data.get("chosen_targets",[])
        soil_tp=data.get("soil_type",-1)
        p1=data.get("prev_crop_1",-1)
        p2=data.get("prev_crop_2",-1)
        p3=data.get("prev_crop_3",-1)
        area_val=data.get("area_ha",-1)

        distinct_crops=data.get("distinct_crops",False)

        N=100
        scenarios=[]
        for _ in range(N):
            # (A) Формируем признаки для классификации
            fclass={}
            if soil_tp!=-1:
                fclass["soil_type"]=float(soil_tp)
            else:
                fclass["soil_type"]=float(random.randint(0,5))

            if p1!=-1:
                fclass["prev_crop_1"]=float(p1)
            else:
                fclass["prev_crop_1"]=float(random.randint(0,6))
            if p2!=-1:
                fclass["prev_crop_2"]=float(p2)
            else:
                fclass["prev_crop_2"]=float(random.randint(0,6))
            if p3!=-1:
                fclass["prev_crop_3"]=float(p3)
            else:
                fclass["prev_crop_3"]=float(random.randint(0,6))

            # area_ha => [area_val-100..area_val+100] если area_val>0
            if area_val>0:
                low=max(1, area_val-1)
                high=min(50000, area_val+1)
                a_ha=random.randint(low,high)
                fclass["area_ha"]=float(a_ha)
            else:
                a_ha=random.randint(10,50000)
                fclass["area_ha"]=float(a_ha)

            fclass["soil_ph"]=round(random.uniform(5,7.5),2)
            fclass["temp_avg"]=round(random.uniform(5,25),1)
            fclass["rainfall_mm"]=float(random.randint(100,800))

            fclass["fert_cost"]=float(random.randint(100,500))
            fclass["fuel_cost"]=float(random.randint(150,600))
            fclass["seed_cost"]=float(random.randint(50,250))
            fclass["market_price"]=float(random.randint(8000,40000))
            fclass["transp_cost"]=float(random.randint(50,1000))
            fclass["num_workers"]=float(random.randint(1,50))

            fclass["humus_percent"]=round(random.uniform(2,10),2)
            fclass["nitrogen_mg_kg"]=round(random.uniform(50,200),2)
            fclass["phosphorus_mg_kg"]=round(random.uniform(10,50),2)
            fclass["potassium_mg_kg"]=round(random.uniform(100,400),2)
            fclass["soil_moisture"]=round(random.uniform(10,40),2)
            fclass["soil_density"]=round(random.uniform(0.8,1.6),2)
            fclass["permeability"]=round(random.uniform(2,15),2)
            fclass["fertile_depth"]=round(random.uniform(15,40),2)
            fclass["salinity"]=float(random.randint(1,3))
            fclass["stoniness"]=float(random.randint(1,3))
            fclass["erosion"]=float(random.randint(1,3))

            # (B) classification -> rec_crop
            df_class=pd.DataFrame([fclass])[FEATURE_NAMES_CLASS]
            rec_crop=classification_model.predict(df_class)[0]
            rec_crop_val=int(rec_crop)

            # (C) regression
            freg={}
            freg["soil_type"]= fclass["soil_type"]
            freg["rec_crop"]= float(rec_crop_val)
            freg["prev_crop_1"]= fclass["prev_crop_1"]
            freg["prev_crop_2"]= fclass["prev_crop_2"]
            freg["prev_crop_3"]= fclass["prev_crop_3"]
            freg["area_ha"]   = fclass["area_ha"]
            freg["soil_ph"]   = fclass["soil_ph"]
            freg["temp_avg"]  = fclass["temp_avg"]
            freg["rainfall_mm"]=fclass["rainfall_mm"]

            freg["fert_cost"]=  fclass["fert_cost"]
            freg["fuel_cost"]=  fclass["fuel_cost"]
            freg["seed_cost"]=  fclass["seed_cost"]
            freg["market_price"]=fclass["market_price"]
            freg["transp_cost"]=fclass["transp_cost"]
            freg["num_workers"]=fclass["num_workers"]

            freg["humus_percent"]=   fclass["humus_percent"]
            freg["nitrogen_mg_kg"]=  fclass["nitrogen_mg_kg"]
            freg["phosphorus_mg_kg"]=fclass["phosphorus_mg_kg"]
            freg["potassium_mg_kg"] =fclass["potassium_mg_kg"]
            freg["soil_moisture"]=   fclass["soil_moisture"]
            freg["soil_density"]=    fclass["soil_density"]
            freg["permeability"]=    fclass["permeability"]
            freg["fertile_depth"]=   fclass["fertile_depth"]
            freg["salinity"]=        fclass["salinity"]
            freg["stoniness"]=       fclass["stoniness"]
            freg["erosion"]=         fclass["erosion"]

            df_reg=pd.DataFrame([freg])[FEATURE_NAMES_REG]

            preds={}
            for tgt,mdl in regression_models.items():
                preds[tgt]=float(mdl.predict(df_reg)[0])

            score_val=0.0
            for ct in chosen_targets:
                if ct in preds:
                    score_val+=preds[ct]

            scenario_dict={
                "soil_type": int(fclass["soil_type"]),
                "prev_crop_1":int(fclass["prev_crop_1"]),
                "prev_crop_2":int(fclass["prev_crop_2"]),
                "prev_crop_3":int(fclass["prev_crop_3"]),
                "area_ha":   int(fclass["area_ha"]),
                "soil_ph":   fclass["soil_ph"],
                "temp_avg":  fclass["temp_avg"],
                "rainfall_mm":fclass["rainfall_mm"],
                "fert_cost": fclass["fert_cost"],
                "fuel_cost": fclass["fuel_cost"],
                "seed_cost": fclass["seed_cost"],
                "market_price": fclass["market_price"],
                "transp_cost": fclass["transp_cost"],
                "num_workers": fclass["num_workers"],
                "humus_percent": fclass["humus_percent"],
                "nitrogen_mg_kg":fclass["nitrogen_mg_kg"],
                "phosphorus_mg_kg":fclass["phosphorus_mg_kg"],
                "potassium_mg_kg": fclass["potassium_mg_kg"],
                "soil_moisture":   fclass["soil_moisture"],
                "soil_density":    fclass["soil_density"],
                "permeability":    fclass["permeability"],
                "fertile_depth":   fclass["fertile_depth"],
                "salinity": int(fclass["salinity"]),
                "stoniness":int(fclass["stoniness"]),
                "erosion":  int(fclass["erosion"]),

                "rec_crop": rec_crop_val,

                # preds (можно не включать, но пусть будут)
                **preds,

                "score": round(score_val,4),
            }
            scenarios.append(scenario_dict)

        # Сортируем по score (desc)
        scenarios_sorted=sorted(scenarios, key=lambda x:x["score"], reverse=True)

        if distinct_crops:
            # (1) Если хотим разные культуры => группируем по rec_crop
            best_by_crop={}
            for scn in scenarios_sorted:
                rc=scn["rec_crop"]
                if rc not in best_by_crop:
                    best_by_crop[rc]=scn
            # Собираем unique_scenarios
            unique_scenarios=list(best_by_crop.values())
            # Сортируем их
            unique_scenarios_sorted=sorted(unique_scenarios, key=lambda x:x["score"], reverse=True)
            topN= unique_scenarios_sorted[:3]
        else:
            # (2) Если режим "как модель предскажет" => просто берем top-3
            topN= scenarios_sorted[:3]

        # Формируем финальный ответ
        final_results=[]
        for scn in topN:
            scn_copy=dict(scn)
            # Удаляем prev_crop_1..3
            for pc in ["prev_crop_1","prev_crop_2","prev_crop_3"]:
                if pc in scn_copy:
                    del scn_copy[pc]
            # Удаляем выбранные цели
            for ct in chosen_targets:
                if ct in scn_copy:
                    del scn_copy[ct]

            # soil_type -> строка
            if "soil_type" in scn_copy:
                st_code=int(scn_copy["soil_type"])
                scn_copy["soil_type"]=soil_mapping.get(st_code,"???")

            # rec_crop -> строка
            if "rec_crop" in scn_copy:
                rc=int(scn_copy["rec_crop"])
                scn_copy["rec_crop"]=crop_mapping.get(rc,"???")

            final_results.append(scn_copy)

        return jsonify({"top_scenarios":final_results}),200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

# ====================== Тест ======================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message":"Dynamic Flask API with distinct crops check is running OK."}),200

# ====================== Запуск ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
