# mes_api_mock.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/rezultat", methods=["POST"])
def primi_podatke():
    data = request.get_json()
    print("📥 Primljeni podaci:", data)
    return jsonify({"status": "OK", "poruka": "Podaci su spremljeni"}), 200

@app.route("/api/zadaci", methods=["GET"])
def daj_zadatke():
    return jsonify([
        {"id": 1, "naziv": "Izračun rešetki"},
        {"id": 2, "naziv": "Provjera podataka"}
    ])

if __name__ == "__main__":
    app.run(port=5001)
