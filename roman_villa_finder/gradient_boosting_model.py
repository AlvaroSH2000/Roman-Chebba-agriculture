import pandas as pd
import numpy as np
from matplotlib.pylab import plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import joblib

# Cargar datos
df = pd.read_excel("..\\data\\processed\\combined_probable_sites_processed.xlsx")

try: df = df.drop(columns=["FID"])
except: df = df.drop(columns=["ID"])

train_data = df.copy()

# 75% of the data is selected
train_df = train_data.sample(frac=0.75, random_state=None)
# it drops the training data
# from the original dataframe
test_df = train_data.drop(train_df.index)

# Quitar ID
X = train_df.drop(columns=["Type"])
y = train_df["Type"]

# Crear el modelo de Gradient Boosting
model = Pipeline(
    steps=[("scaler", StandardScaler()), ("model", GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42))]
)


# Validación amb les dades d'entrenament
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("Accuracy CV media:", scores.mean())
print("Accuracy CV std:", scores.std())

# Ajuste final
model.fit(X, y)


y_test = model.predict(df.drop(columns=["Type"]))
accuracy_test = (y_test == df["Type"]).mean() * 100 
print(f"Accuracy en test: {accuracy_test:.2f}%")

# Guardar el modelo
model_path = ".\\models\\combined_probable_trained_model.joblib"
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")