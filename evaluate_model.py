import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Učitaj podatke
df = pd.read_csv("lamela_podaci_s_tipom.csv")

# Priprema rezultata
evaluacija = []

# Grupiraj po tipu i broju lamela
for (tip, lamela), grupa in df.groupby(["Tip", "Broj lamela"]):
    X = grupa[["H (mm)"]].values
    y = grupa["Raster (a) (mm)"].values

    try:
        reg_path = f"regresori/{tip}_reg_{lamela}.joblib"
        scaler_path = f"regresori/{tip}_scaler_{lamela}.joblib"

        reg = joblib.load(reg_path)
        scaler = joblib.load(scaler_path)

        X_scaled = scaler.transform(X)
        y_pred = reg.predict(X_scaled)

        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        evaluacija.append({
            "Tip": tip,
            "Broj lamela": lamela,
            "Broj uzoraka": len(grupa),
            "MAE": round(mae, 3),
            "R²": round(r2, 4)
        })

    except Exception as e:
        evaluacija.append({
            "Tip": tip,
            "Broj lamela": lamela,
            "Broj uzoraka": len(grupa),
            "MAE": "ERR",
            "R²": "ERR"
        })

# Spremi u DataFrame i TXT
eval_df = pd.DataFrame(evaluacija)
eval_df = eval_df.sort_values(by=["Tip", "Broj lamela"])

eval_df.to_csv("evaluacija_modela.csv", index=False)

with open("evaluacija_modela.txt", "w") as f:
    f.write("Evaluacija modela po tipu i broju lamela:\n\n")
    for _, row in eval_df.iterrows():
        f.write(f"Tip: {row['Tip']}, Lamela: {row['Broj lamela']}, MAE: {row['MAE']}, R²: {row['R²']}, Uzoraka: {row['Broj uzoraka']}\n")

print("✅ Evaluacija završena. Rezultati spremljeni u evaluacija_modela.txt i evaluacija_modela.csv")
