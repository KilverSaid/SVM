import joblib
import pandas as pd

modelo = joblib.load("models/kmeans_riesgo_actuarial.pkl")

cliente = pd.DataFrame([{
    "age": 45,
    "sex": "male",
    "bmi": 31.2,
    "children": 2,
    "smoker": "yes",
    "region": "southeast",
    "charges": 28000
}])

cluster = modelo.predict(cliente)[0]
