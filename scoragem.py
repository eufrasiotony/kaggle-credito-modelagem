

import pandas as pd


def score_dataset(df_score, caminho_pkl):
    import pickle
    import pandas as pd

    # -----------------------------------
    # LOAD ARTEFATOS
    # -----------------------------------
    with open(caminho_pkl, "rb") as f:
        artefatos = pickle.load(f)

    model = artefatos["modelo"]
    features = artefatos["features"]
    bins_dict = artefatos["bins"]
    woe_maps = artefatos["woe_maps"]

    X_woe = pd.DataFrame(index=df_score.index)

    # ===================================
    # NUMÉRICAS (SEM from_tuples)
    # ===================================
    for var, bins in bins_dict.items():
        X_woe[var] = (
            pd.cut(df_score[var], bins=bins)
            .map(woe_maps["numericas"][var])
            .astype(float)
        )

    # ===================================
    # CATEGÓRICAS
    # ===================================
    for var, woe_map in woe_maps["categoricas"].items():
        X_woe[var] = df_score[var].map(woe_map).astype(float)

    # ===================================
    # FINAL
    # ===================================
    X_woe = X_woe.fillna(0)
    X_woe = X_woe.reindex(columns=features, fill_value=0)

    df_score = df_score.copy()
    df_score["pd"] = model.predict_proba(X_woe)[:, 1]

    return df_score



# =========================================================
# USO DA FUNÇÃO (SÓ DEPOIS!)
# =========================================================


df_novo = pd.read_csv("credit_risk_dataset.csv")


df_scored = score_dataset(
    df_score=df_novo,
    caminho_pkl=r"c:\Users\Tony Germano\Documents\GitHub\kaggle-credito-modelagem\modelo_credito_woe.pkl"
)

print(df_scored.head())

# Salvando o resultado em um novo arquivo CSV
df_scored.to_csv("credit_risk_dataset_scored.csv", index=False)