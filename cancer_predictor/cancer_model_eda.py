# 1. Importacion librerias



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from pydantic import BaseModel
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ## EDA

df = pd.read_csv('./patients.csv')
df.head()





# # AGE
#

age_range = df.copy()
range_ = age_range.groupby(['age']).size().reset_index(name='total_patients')
range_.sort_values(by = 'total_patients', ascending = False, inplace = False)

# +
sns.set()
plt.subplots(figsize=(20,7))

age_data = sns.barplot(x = 'age', y = 'total_patients', data = range_)
age_data.set_xticklabels(age_data.get_xticklabels(),
                         rotation=90,
                         fontdict={'fontsize':10})
age_data.set_title('Patients by age range', size=20)
# -

pd.crosstab(index=age_range['irradiat'],
            columns=age_range['age'], margins=True)

pd.crosstab(index=age_range['menopause'],
            columns=age_range['age'], margins=True)

# # Tumor Size

tumor_size = df.copy()
tumor_ = tumor_size.groupby(['tumor_size']).size().reset_index(name='total_patients')
# tumor_.sort_values(by = 'total_patients', ascending = False, inplace = False)
tumor_

# +
# sns.set()
# plt.subplots(figsize=(20,7))

tumor_data = sns.barplot(x = 'tumor_size', y = 'total_patients', data = tumor_)
tumor_data.set_xticklabels(tumor_data.get_xticklabels(),
                         rotation=90,
                         fontdict={'fontsize':10})
tumor_data.set_title('Patients by tumor size', size=20)
# -

pd.crosstab(index=tumor_size['irradiat'],
            columns=tumor_size['tumor_size'], margins=True)

pd.crosstab(index=tumor_size['irradiat'], columns=tumor_size['tumor_size'],
            margins=True).apply(lambda r: round(r/len(tumor_size) *100,2),
                                axis=1)

# # Deg malig

deg_malig = df.copy()
deg = deg_malig.groupby(['deg_maling']).size().reset_index(name='total_patients')
# tumor_.sort_values(by = 'total_patients', ascending = False, inplace = False)
deg

deg_data = sns.barplot(x = 'deg_maling', y = 'total_patients', data = deg)
deg_data.set_xticklabels(deg_data.get_xticklabels(),
                         rotation=90,
                         fontdict={'fontsize':10})
deg_data.set_title('Patients by deg malig', size=20)

pd.crosstab(index=deg_malig['irradiat'],
            columns=deg_malig['deg_maling'], margins=True)

# ## Menopause

menopause_ = df.copy()
menopause = menopause_.groupby(['menopause']).size().reset_index(name='total_patients')
# tumor_.sort_values(by = 'total_patients', ascending = False, inplace = False)
menopause 

pd.crosstab(index=menopause_['irradiat'],
            columns=menopause_['menopause'], margins=True)

# ## class_

class_ = df.copy()
class_data = class_.groupby(['class_']).size().reset_index(name='total_patients')
# tumor_.sort_values(by = 'total_patients', ascending = False, inplace = False)
class_data

pd.crosstab(index=class_['irradiat'],
            columns=class_['class_'], margins=True)

pd.crosstab(index=class_['irradiat'],
            columns=class_['class_'],
            margins=True).apply(lambda r: round(r/len(class_) *100,2),
                                axis=1)

# 2. Clase que hereda BaseModel con caracteristicas de los pacientes






df_train, df_test = train_test_split(df, test_size = 0.3)


def dummies_model(df, df_prep):
    
    x = df.drop('irradiat', axis=1)
    
    data_preproccesing = pd.get_dummies(x)
    data_columns = data_preproccesing.columns
    
    df_model = pd.DataFrame(columns=data_columns)
    
    x_model = df_prep.drop('irradiat', axis=1)
    y_model = df_prep['irradiat']
    
    data_preproccesing = pd.get_dummies(x_model)
    data = pd.concat([df_model, data_preproccesing], axis=0).fillna(0)
    
    return data, y_model


rfc = RandomForestClassifier()
model = rfc.fit(dummies_model(df, df_train)[0], dummies_model(df, df_train)[1])

df_train['y_predict'] = model.predict(dummies_model(df, df_train)[0])
df_train.head()

confusion_matrix(df_train['irradiat'], df_train['y_predict'])

accuracy_score(df_train['irradiat'], df_train['y_predict'])

df_test['y_predict'] = model.predict(dummies_model(df, df_test)[0])
df_test.head()



confusion_matrix(df_test['irradiat'], df_test['y_predict'])

accuracy_score(df_test['irradiat'], df_test['y_predict'])








