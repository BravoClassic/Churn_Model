from fastapi import FastAPI
import pickle
app = FastAPI()
import pandas as pd

with open("ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)



def preprocess_data(customer_data: dict):
    input_data = {
        "NumOfProducts": customer_data["NumOfProducts"],
        "IsActiveMember": int( customer_data["IsActiveMember"]),
        "Age": customer_data["Age"],
        "Geography_Germany": 1 if customer_data["Geography_Germany"] == "Germany" else 0,
        "Balance": customer_data["Balance"],
        "Geography_France": 1 if customer_data["Geography_France"] == "France" else 0,
        "Gender_Female": 1 if customer_data["Gender_Female"] == "Female" else 0,
        "Geography_Spain": 1 if customer_data["Geography_Spain"] == "Spain" else 0,
        "CreditScore": customer_data["CreditScore"],
        "EstimatedSalary": customer_data["EstimatedSalary"],
        "HasCrCard": int(customer_data["HasCrCard"]),
        "Tenure": customer_data["Tenure"],
        "Gender_Male": 1 if customer_data["Gender_Male"] == "Male" else 0,
    }

    customer_data_df = pd.DataFrame([input_data])
    print(customer_data_df)
    return customer_data_df

def get_prediction(customer_data: dict):
    preprocess_data_df = preprocess_data(customer_data)
    prediction = model.predict(preprocess_data_df)
    probability = model.predict_proba(preprocess_data_df)
    return prediction, probability

@app.post("/predict")
def predict(customer_data: dict):
    prediction, probability = get_prediction(customer_data)
    return {"prediction": prediction.tolist(), "probability": probability.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)