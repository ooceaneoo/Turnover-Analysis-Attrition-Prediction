import pandas as pd


def load_raw_data(path_sirh, path_eval, path_sondage):
    df_sirh = pd.read_csv(path_sirh)
    df_eval = pd.read_csv(path_eval)
    df_sondage = pd.read_csv(path_sondage)
    return df_sirh, df_eval, df_sondage