# 1. Importacion librerias

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from pydantic import BaseModel
import joblib

# 2. Clase que hereda BaseModel con caracteristicas de los pacientes


class PatientCharacteristics(BaseModel):
    class_: str
    age: str
    menopause: str
    tumor_size: str
    inv_nodes: str
    node_caps: str
    deg_maling: float
    breast: str
    breast_quad: str


# 3. Clase para entrenar modelo y hacer predicciones
class CancerModel:
    # 6. constructor de clases, cargar conjunto de datos y modelo si existe
    # si no llama metodo _train_model y guarda el modelo
    def __init__(self):
        self.df = pd.read_csv('./cancer_predictor/patients.csv')
        self.model_fname_ = 'cancer_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    # 4. Metodo train model con random forest (entrenamiento del modelo).
    def _train_model(self):
        x = self.df.drop('irradiat', axis=1)
        y = self.df['irradiat']

        data_preproccesing = pd.get_dummies(x)

        data_columns = data_preproccesing.columns

        rfc = RandomForestClassifier()
        model = rfc.fit(data_preproccesing, y)
        return model, data_columns

    # 5. Prediccion apartir de los parametros ingresados y devuelve pred y prob
    def predict_cancer(self, class_, age, menopause, tumor_size, inv_nodes,
                       node_caps, deg_maling, breast, breast_quad):

        df = pd.DataFrame(columns=self.model[1])

        data_in = {"class_": class_,
                   "age": age,
                   "menopause": menopause,
                   "tumor_size": tumor_size,
                   "inv_nodes": inv_nodes,
                   "node_caps": node_caps,
                   "deg_maling": deg_maling,
                   "breast": breast,
                   "breast_quad": breast_quad
                   }
        data_df = pd.DataFrame([data_in], columns=data_in.keys())

        data_predict = pd.get_dummies(data_df)

        patients = pd.concat([df, data_predict], axis=0).fillna(0)

        prediction = self.model[0].predict(patients)
        probability = self.model[0].predict_proba(patients).max()
        return prediction[0], probability
