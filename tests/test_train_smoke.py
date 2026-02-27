import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from turnover_ml.features import add_engineered_features


def test_training_pipeline_smoke_runs():
    """
    Smoke test = vérifie que le pipeline complet
    (preprocess + undersampling + modèle) s'entraîne sans planter.
    """
    X = pd.DataFrame({
        "revenu_mensuel": [3000, 4000, 3500, 4200],
        "genre": ["f", "m", "f", "m"],
        "annees_dans_le_poste_actuel": [1, 2, 1, 3],
        "annees_dans_l_entreprise": [3, 5, 4, 6],
        "annees_depuis_la_derniere_promotion": [1, 2, 1, 3],
        "annee_experience_totale": [5, 8, 6, 10],
        "satisfaction_employee_environnement": [2, 4, 3, 5],
        "satisfaction_employee_nature_travail": [3, 4, 3, 5],
        "satisfaction_employee_equipe": [2, 5, 3, 5],
        "satisfaction_employee_equilibre_pro_perso": [3, 4, 3, 5],
    })
    y = pd.Series([1, 0, 0, 0])

    X = add_engineered_features(X)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("under", RandomUnderSampler(random_state=42)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape[0] == X.shape[0]