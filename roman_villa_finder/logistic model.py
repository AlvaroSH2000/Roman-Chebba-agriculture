import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Cargar datos
df = pd.read_excel("..\\data\\processed\\combined_sites_processed.xlsx")

# train_data = df.drop(columns=["FID"])
train_data = df.copy
# 75% of the data is selected
train_df = train_data.sample(frac=0.75, random_state=4)
# it drops the training data
# from the original dataframe
test_df = train_data.drop(train_df.index)

# Quitar ID
X = train_df.drop(columns=["Type"])
y = train_df["Type"]

# Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l1",      # cambia a "l1" si quieres selección de variables
        solver="liblinear",
        max_iter=100
    ))
])

# Validación
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("Accuracy CV media:", scores.mean())
print("Accuracy CV std:", scores.std())

# Ajuste final
model.fit(X, y)

# Extraer pesos
coefs = model.named_steps["clf"].coef_[0]

importance = pd.DataFrame({
    "feature": X.columns,
    "coef": coefs,
    "abs_coef": np.abs(coefs),
    "odds_ratio": np.exp(coefs)
}).sort_values("abs_coef", ascending=False)

print(importance)

y_test = model.predict(test_df.drop(columns=["Type"]))
accuracy_test = (y_test == test_df["Type"]).mean() * 100 
print(f"Accuracy en test: {accuracy_test:.2f}%")
