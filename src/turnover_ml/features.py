def add_custom_features(X):
    """
    Add engineered features to dataframe X.
    """
    X = X.copy()

    if "annees_dans_le_poste_actuel" in X.columns and "annees_dans_l_entreprise" in X.columns:
        X["ratio_anciennete_poste"] = (
            X["annees_dans_le_poste_actuel"] / (X["annees_dans_l_entreprise"] + 1)
        )

    return X