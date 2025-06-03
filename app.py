from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            tip = request.form["tip"]
            H = float(request.form["H"])
            B = float(request.form["B"])

            if tip == "FŽ":
                original_lamela = 1 if H <= 103 else int((H - 104) // 75) + 2
                prikaz_lamela = original_lamela * 2 if B > 1199 else original_lamela

                reg = joblib.load(f"regresori/{tip}_reg_{original_lamela}.joblib")
                sc = joblib.load(f"regresori/{tip}_scaler_{original_lamela}.joblib")
                raster = reg.predict(sc.transform(np.array([[H]])))[0]

            elif tip == "PŽ-S":
                original_lamela = 1 if H <= 124 else int((H - 125) // 70) + 2
                prikaz_lamela = original_lamela * 2 if B > 999 else original_lamela

                reg = joblib.load(f"regresori/{tip}_reg_{original_lamela}.joblib")
                sc = joblib.load(f"regresori/{tip}_scaler_{original_lamela}.joblib")
                raster = reg.predict(sc.transform(np.array([[H]])))[0]

            elif tip == "PŽ-T":
                if H <= 112:
                    lamela_base = 1
                else:
                    lamela_base = int((H - 113) // 75) + 2

                raster = round(H / lamela_base,1)

                if B > 999:
                    lamela = lamela_base * 2
                else:
                    lamela = lamela_base

                prediction = f"Broj lamela: {lamela}, Raster: {round(raster, 2)} mm"
                return render_template("index.html", prediction=prediction)

            else:
                raise ValueError("Nepoznat tip lamele.")

            prediction = f"Broj lamela: {prikaz_lamela}, Raster: {round(raster, 2)} mm"

        except Exception as e:
            prediction = f"Greška: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
