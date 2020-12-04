# 1. Library imports

import uvicorn
import pandas as pd
from fastapi import FastAPI
from cancer_predictor.cancer_model import PatientCharacteristics, CancerModel

# 2. Creacion app object and model object

app = FastAPI()
model = CancerModel()

# 3. Se expone funcionalidad de predicion, hace prediccion
# a partir de datos en tipo JSON y retorna pred y prob


@app.post('/predict')
def predict_cancer(patients: PatientCharacteristics):
    data = patients.dict()

    prediction, probability = model.predict_cancer(
        data['class_'], data['age'], data['menopause'], data['tumor_size'],
        data['inv_nodes'], data['node_caps'], data['deg_maling'], data['breast'], data['breast_quad']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

# 4. Correr Api con uvicorn en http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

