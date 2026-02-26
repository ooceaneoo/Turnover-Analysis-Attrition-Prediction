import pandas as pd

SATISFACTION_COLS = [
    "satisfaction_employee_environnement",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "annees_dans_le_poste_actuel" in X.columns and "annees_dans_l_entreprise" in X.columns:
        X["ratio_anciennete_poste"] = X["annees_dans_le_poste_actuel"] / (X["annees_dans_l_entreprise"] + 1)

    if all(col in X.columns for col in SATISFACTION_COLS):
        X["score_satisfaction_global"] = X[SATISFACTION_COLS].mean(axis=1)

    if "annees_dans_l_entreprise" in X.columns and "annees_depuis_la_derniere_promotion" in X.columns:
        X["stagnation_carriere"] = X["annees_dans_l_entreprise"] - X["annees_depuis_la_derniere_promotion"]

    if "revenu_mensuel" in X.columns and "annee_experience_totale" in X.columns:
        X["salaire_par_annee_experience"] = X["revenu_mensuel"] / (X["annee_experience_totale"] + 1)

    return X