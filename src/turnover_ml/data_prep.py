import pandas as pd


def load_and_merge_data(path_sirh, path_eval, path_sondage) -> pd.DataFrame:
    df_sirh = pd.read_csv(path_sirh)
    df_eval = pd.read_csv(path_eval)
    df_sondage = pd.read_csv(path_sondage)

    # Extraction id_employee depuis eval_number
    df_eval = df_eval.copy()
    df_eval["id_employee"] = df_eval["eval_number"].str.replace("E_", "", regex=False).astype(int)
    df_eval = df_eval.drop(columns=["eval_number"], errors="ignore")

    # Harmoniser sondage
    df_sondage = df_sondage.copy()
    if "code_sondage" in df_sondage.columns:
        df_sondage = df_sondage.rename(columns={"code_sondage": "id_employee"})

    df_temp = df_sirh.merge(df_sondage, on="id_employee", how="inner")
    df_merged = df_temp.merge(df_eval, on="id_employee", how="inner")
    return df_merged


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target binaire
    if "a_quitte_l_entreprise" in df.columns:
        df["a_quitte_l_entreprise_num"] = df["a_quitte_l_entreprise"].apply(lambda x: 1 if x == "Oui" else 0)

    # Pourcentage -> float
    if "augementation_salaire_precedente" in df.columns:
        def _nettoyer_pourcentage(x):
            if isinstance(x, str):
                return float(x.replace(" %", "").strip())
            return None

        df["augmentation_salaire_num"] = df["augementation_salaire_precedente"].apply(_nettoyer_pourcentage)
        df = df.drop(columns=["augementation_salaire_precedente"], errors="ignore")

    # Nettoyage des colonnes object
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace("  ", " ")

    # Drop colonnes constantes
    constantes = df.columns[df.nunique(dropna=False) <= 1]
    df = df.drop(columns=constantes, errors="ignore")

    return df


def build_xy(df: pd.DataFrame):
    if "a_quitte_l_entreprise_num" not in df.columns:
        raise ValueError("Target column a_quitte_l_entreprise_num not found.")

    y = df["a_quitte_l_entreprise_num"].astype(int)
    X = df.drop(columns=["id_employee", "a_quitte_l_entreprise", "a_quitte_l_entreprise_num"], errors="ignore")

    # Drop colonnes constantes dans X
    cols_unique = X.columns[X.nunique(dropna=False) <= 1]
    X = X.drop(columns=cols_unique, errors="ignore")

    return X, y