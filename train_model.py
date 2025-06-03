import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# Učitaj objedinjeni CSV
df = pd.read_csv("lamela_podaci_s_tipom.csv")

# Treniraj regresore po tipu i broju lamela
os.makedirs("regresori", exist_ok=True)
greske = {}
#Razlaganje učenja na onoliko modela koliko ima kombinacija tipa i broja lamele
for (tip, lamela), grupa in df.groupby(["Tip", "Broj lamela"]):
    X = grupa[["H (mm)"]].values
    y = grupa["Raster (a) (mm)"].values
    #skaliranje podataka kako bi model bolje učio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #učenje modela
    reg = LinearRegression()
    reg.fit(X_scaled, y)
    #spremanje naučenog (a i b) kako bi ih kasnije mogli koristiti
    joblib.dump(reg, f"regresori/{tip}_reg_{lamela}.joblib")
    joblib.dump(scaler, f"regresori/{tip}_scaler_{lamela}.joblib")
    #ocijena modela
    y_pred = reg.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    greske[f"{tip}_{lamela}"] = mae

# Spremi evaluaciju
with open("evaluacija_regresora.txt", "w") as f:
    f.write("MAE po tipu i broju lamela:\n")
    for k in sorted(greske):
        f.write(f"{k}: MAE = {greske[k]:.3f} mm\n")
#provjera jeli sve prošlo ok
print("✅ Treniranje završeno.")
