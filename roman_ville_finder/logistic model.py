import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Cargar datos
df = pd.read_excel("..\\data\\processed\\other_sites_processed.xlsx")

# Quitar ID
X = df.drop(columns=["FID", "Type"])
y = df["Type"]

# Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",      # cambia a "l1" si quieres selección de variables
        solver="liblinear",
        max_iter=1000
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