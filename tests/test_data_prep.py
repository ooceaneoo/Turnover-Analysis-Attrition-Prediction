import pandas as pd

from turnover_ml.data_prep import load_and_merge_data, clean_dataset, build_xy


def test_load_and_merge_data_returns_dataframe(tmp_path):
    """
    Test unitaire: vérifie que la fonction de merge
    lit des CSV et renvoie un DataFrame cohérent.
    """
    sirh = pd.DataFrame({
        "id_employee": [1, 2],
        "a_quitte_l_entreprise": ["Oui", "Non"],
        "revenu_mensuel": [3000, 4000],
    })
    evalf = pd.DataFrame({
        "eval_number": ["E_1", "E_2"],
        "note_eval": [3, 4],
    })
    sondage = pd.DataFrame({
        "code_sondage": [1, 2],
        "satisfaction_employee_environnement": [2, 4],
    })

    sirh_path = tmp_path / "extrait_sirh.csv"
    eval_path = tmp_path / "extrait_eval.csv"
    sondage_path = tmp_path / "extrait_sondage.csv"
    sirh.to_csv(sirh_path, index=False)
    evalf.to_csv(eval_path, index=False)
    sondage.to_csv(sondage_path, index=False)

    df = load_and_merge_data(sirh_path, eval_path, sondage_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert "id_employee" in df.columns


def test_clean_dataset_creates_binary_target():
    """
    Test unitaire: vérifie que clean_dataset crée bien la target binaire.
    """
    df = pd.DataFrame({
        "id_employee": [1, 2],
        "a_quitte_l_entreprise": ["Oui", "Non"],
    })
    out = clean_dataset(df)

    assert "a_quitte_l_entreprise_num" in out.columns
    assert out["a_quitte_l_entreprise_num"].tolist() == [1, 0]

def test_build_xy_splits_X_y():
    """
    Test unitaire: vérifie que build_xy renvoie bien X et y,
    et que X ne contient pas la target.
    """
    df = pd.DataFrame({
        "id_employee": [1, 2],
        "a_quitte_l_entreprise": ["Oui", "Non"],
        "a_quitte_l_entreprise_num": [1, 0],
        "revenu_mensuel": [3000, 4000],
    })

    X, y = build_xy(df)

    assert "revenu_mensuel" in X.columns
    assert "a_quitte_l_entreprise_num" not in X.columns
    assert y.shape[0] == 2