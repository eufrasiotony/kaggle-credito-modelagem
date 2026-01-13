# =========================================================
# MODELAGEM DE CRÉDITO - REGRESSÃO LOGÍSTICA COM WOE
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =========================================================
# FUNÇÕES
# =========================================================

def binning_numerico(df, var, q=10):
    serie = df[var].copy().fillna(df[var].median())

    if serie.nunique() <= 1:
        return None

    if serie.nunique() < q:
        return pd.cut(serie, bins=serie.nunique())

    try:
        return pd.qcut(serie, q=q, duplicates="drop")
    except ValueError:
        return pd.cut(serie, bins=q)


def calcula_woe(df, feature, target):
    tab = (
        df.groupby(feature, observed=True)[target]
        .agg(count="count", bad="sum")
    )

    tab["good"] = tab["count"] - tab["bad"]

    tab["bad"] = tab["bad"].replace(0, 0.5)
    tab["good"] = tab["good"].replace(0, 0.5)

    tab["bad_dist"] = tab["bad"] / tab["bad"].sum()
    tab["good_dist"] = tab["good"] / tab["good"].sum()

    tab["woe"] = np.log(tab["good_dist"] / tab["bad_dist"])
    tab["woe"] = tab["woe"].replace([np.inf, -np.inf], 0)

    return tab["woe"].to_dict()


def calcula_ks(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = pd.qcut(df["score"], 10, duplicates="drop")

    grouped = (
        df.groupby("decile", observed=True)
        .agg(bad=("y", "sum"), total=("y", "count"))
    )

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["cum_bad"] = (grouped["bad"] / grouped["bad"].sum()).cumsum()
    grouped["cum_good"] = (grouped["good"] / grouped["good"].sum()).cumsum()

    return (grouped["cum_bad"] - grouped["cum_good"]).abs().max()


def calcula_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, buckets + 1)
    )

    exp_dist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_dist = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (act_dist - exp_dist) *
        np.log((act_dist + 1e-6) / (exp_dist + 1e-6))
    )

    return psi


def calcula_iv(df_woe, y):
    ivs = {}

    for col in df_woe.columns:
        tmp = pd.DataFrame({"woe": df_woe[col], "y": y})

        grouped = (
            tmp.groupby("woe", observed=True)
            .agg(bad=("y", "sum"), total=("y", "count"))
        )

        grouped["good"] = grouped["total"] - grouped["bad"]

        grouped["bad_dist"] = grouped["bad"] / grouped["bad"].sum()
        grouped["good_dist"] = grouped["good"] / grouped["good"].sum()

        iv = np.sum(
            (grouped["good_dist"] - grouped["bad_dist"]) *
            np.log(
                (grouped["good_dist"] + 1e-6) /
                (grouped["bad_dist"] + 1e-6)
            )
        )

        ivs[col] = iv

    return pd.DataFrame.from_dict(ivs, orient="index", columns=["IV"])


def checa_monotonicidade(df, var_bin, target):
    tab = (
        df.groupby(var_bin, observed=True)[target]
        .mean()
        .reset_index()
    )

    diffs = tab[target].diff().dropna()
    monotonic = diffs.ge(0).all() or diffs.le(0).all()

    return tab, monotonic


# =========================================================
# LEITURA DOS DADOS
# =========================================================

df = pd.read_csv("credit_risk_dataset.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================================================
# SEPARAÇÃO DE VARIÁVEIS
# =========================================================

num_vars = X_train.select_dtypes(exclude="object").columns.tolist()
cat_vars = X_train.select_dtypes(include="object").columns.tolist()

X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)

bin_vars = []

# =========================================================
# BINNING + WOE NUMÉRICAS
# =========================================================

for var in num_vars:
    bins_serie = binning_numerico(X_train, var)

    if bins_serie is None:
        continue

    bin_col = f"{var}_bin"
    bin_vars.append(bin_col)

    X_train[bin_col] = bins_serie
    X_test[bin_col] = pd.cut(
        X_test[var],
        bins=bins_serie.cat.categories
    )

    tmp = pd.concat([X_train[[bin_col]], y_train], axis=1)
    woe_map = calcula_woe(tmp, bin_col, "loan_status")

    X_train_woe[var] = X_train[bin_col].map(woe_map)
    X_test_woe[var] = X_test[bin_col].map(woe_map)

# =========================================================
# WOE CATEGÓRICAS
# =========================================================

for var in cat_vars:
    tmp = pd.concat([X_train[[var]], y_train], axis=1)
    woe_map = calcula_woe(tmp, var, "loan_status")

    X_train_woe[var] = X_train[var].map(woe_map)
    X_test_woe[var] = X_test[var].map(woe_map)

# =========================================================
# TRATAMENTO FINAL
# =========================================================

X_train_woe = X_train_woe.astype(float).fillna(0)
X_test_woe = X_test_woe.astype(float).fillna(0)

const_cols = X_train_woe.columns[X_train_woe.nunique() <= 1]
X_train_woe.drop(columns=const_cols, inplace=True)
X_test_woe.drop(columns=const_cols, inplace=True)

# =========================================================
# MODELO
# =========================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_woe, y_train)

# =========================================================
# OVERFITTING
# =========================================================

proba_train = model.predict_proba(X_train_woe)[:, 1]
proba_test = model.predict_proba(X_test_woe)[:, 1]

print("\n--- OVERFITTING ---")
print(f"AUC Train: {roc_auc_score(y_train, proba_train):.4f}")
print(f"AUC Test : {roc_auc_score(y_test, proba_test):.4f}")
print(f"KS  Train: {calcula_ks(y_train, proba_train):.4f}")
print(f"KS  Test : {calcula_ks(y_test, proba_test):.4f}")

# =========================================================
# PSI
# =========================================================

psi = calcula_psi(proba_train, proba_test)
print(f"\nPSI Score: {psi:.4f}")

# =========================================================
# IV
# =========================================================

iv_table = calcula_iv(X_train_woe, y_train)

print("\n--- IV ---")
print(iv_table.sort_values("IV", ascending=False))

# =========================================================
# MONOTONICIDADE
# =========================================================

print("\n--- MONOTONICIDADE ---")

for bin_var in bin_vars:
    tab, mono = checa_monotonicidade(
        pd.concat([X_train[[bin_var]], y_train], axis=1),
        bin_var,
        "loan_status"
    )
    print(f"{bin_vars}: Monotônica = {mono}")

# =========================================================
# GERAÇÃO DO ARQUIVO .PKL (ARTEFATOS DO MODELO)
# =========================================================

import pickle

artefatos = {
    # -----------------------------------------------------
    # MODELO
    # -----------------------------------------------------
    "modelo": model,

    # -----------------------------------------------------
    # FEATURES FINAIS (ORDEM IMPORTA)
    # -----------------------------------------------------
    "features": X_train_woe.columns.tolist(),

    # -----------------------------------------------------
    # BINS NUMÉRICOS (OBRIGATÓRIO PARA SCORING)
    # chave = nome da variável original
    # valor = IntervalIndex dos bins
    # -----------------------------------------------------
    "bins": {
        var: X_train[f"{var}_bin"].cat.categories
        for var in num_vars
        if f"{var}_bin" in X_train.columns
    },

    # -----------------------------------------------------
    # MAPAS DE WOE
    # -----------------------------------------------------
    "woe_maps": {
        "numericas": {
            var: calcula_woe(
                pd.concat(
                    [X_train[[f"{var}_bin"]], y_train],
                    axis=1
                ),
                f"{var}_bin",
                "loan_status"
            )
            for var in num_vars
            if f"{var}_bin" in X_train.columns
        },
        "categoricas": {
            var: calcula_woe(
                pd.concat([X_train[[var]], y_train], axis=1),
                var,
                "loan_status"
            )
            for var in cat_vars
        }
    },

    # -----------------------------------------------------
    # MÉTRICAS (AUDITORIA)
    # -----------------------------------------------------
    "metricas": {
        "auc_train": roc_auc_score(y_train, proba_train),
        "auc_test": roc_auc_score(y_test, proba_test),
        "ks_train": calcula_ks(y_train, proba_train),
        "ks_test": calcula_ks(y_test, proba_test),
        "psi": psi
    },

    # -----------------------------------------------------
    # INFORMATION VALUE
    # -----------------------------------------------------
    "iv": iv_table
}

# =========================================================
# SALVAR ARQUIVO
# =========================================================

with open("modelo_credito_woe.pkl", "wb") as f:
    pickle.dump(artefatos, f)

print("\n✅ Arquivo 'modelo_credito_woe.pkl' gerado com sucesso!")
