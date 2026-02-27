import pandas as pd
from turnover_ml.features import add_engineered_features


def test_add_engineered_features_adds_expected_columns():
    """
    Test unitaire: vérifie que les features engineered sont ajoutées.
    """
    X = pd.DataFrame({
        "annees_dans_le_poste_actuel": [2, 3],
        "annees_dans_l_entreprise": [5, 6],
        "annees_depuis_la_derniere_promotion": [1, 2],
        "revenu_mensuel": [3000, 4000],
        "annee_experience_totale": [10, 8],
        "satisfaction_employee_environnement": [2, 4],
        "satisfaction_employee_nature_travail": [3, 4],
        "satisfaction_employee_equipe": [2, 5],
        "satisfaction_employee_equilibre_pro_perso": [3, 4],
    })

    out = add_engineered_features(X)

    assert "ratio_anciennete_poste" in out.columns
    assert "score_satisfaction_global" in out.columns
    assert "stagnation_carriere" in out.columns
    assert "salaire_par_annee_experience" in out.columns