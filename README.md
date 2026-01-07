# 📊 Modelagem de Crédito com Regressão Logística e WOE

Este projeto implementa um **modelo completo de crédito** utilizando **Regressão Logística com Weight of Evidence (WOE)**, seguindo **boas práticas de mercado bancário**. O pipeline contempla desde o tratamento de dados até métricas avançadas de validação, garantindo **interpretabilidade e auditabilidade**. Os dados desse projeto foram fornecidos por https://www.kaggle.com/.

---

## 🎯 Objetivo

Construir um modelo de **classificação binária (inadimplência)** capaz de:

* Separar bons e maus pagadores com alta discriminação
* Ser interpretável (scorecard-ready)
* Atender critérios de validação exigidos por bancos e comitês de risco

---

## 📁 Estrutura do Projeto

```
├── import.py                   # Download do arquivo para scoragem
├── credit_risk_dataset.csv      # Base de Dados baixada do Kaggle
├── woe_regressao.py              # Script principal do modelo
├── scoragem.py                 # Processo para scorar com o modelo que foi criado no woe_regressao.py
├── credit_risk_dataset_scored.csv        # Base de Dados baixada do Kaggle com score (somente para teste)
├── README.md                     # Documentação do projeto
```

---

## 🧠 Metodologia Utilizada

### 1. Split da Base

* Treino: 70%
* Teste: 30%
* Estratificação pelo target (`loan_status`)

### 2. Binning

* Variáveis numéricas: `qcut` (quantis)
* Tratamento de missing values
* Tratamento de variáveis constantes

### 3. Weight of Evidence (WOE)

* Cálculo por faixa (bins ou categorias)
* Smoothing para evitar WOE infinito
* Transformação consistente treino/teste

### 4. Modelo

* Regressão Logística
* Variáveis transformadas em WOE
* Alta interpretabilidade

---

## 📈 Métricas de Performance

### 🔹 AUC (ROC)

Avalia a capacidade de discriminação do modelo.

Referência:

* < 0.70: fraco
* 0.70 – 0.80: bom
* 0.80 – 0.85: muito bom
* > 0.85: excelente

* AUC Train: 0.8758
* AUC Test : 0.8773

### 🔹 KS (Kolmogorov-Smirnov)

Mede a separação máxima entre bons e maus pagadores.

Referência:

* < 0.30: fraco
* 0.30 – 0.40: aceitável
* 0.40 – 0.50: bom
* > 0.50: excelente

* KS  Train: 0.6110
* KS  Test : 0.6075
---

## 🧪 Validações Avançadas

### 1️⃣ Overfitting

Comparação entre métricas de treino e teste:

* AUC Train vs Test
* KS Train vs Test

Critério:

* Diferença ≤ 5 p.p. → saudável

| Métrica | Train   | Test    | Overfitting (%) |
|---------|---------|--------|----------------|
| AUC     | 0.8758  | 0.8773 | -0.17          |
| KS      | 0.6110  | 0.6075 | 0.57           |
---

### 2️⃣ PSI – Population Stability Index

Avalia a **estabilidade do score** entre treino e teste.

Referência:

* PSI < 0.10 → estável
* 0.10 – 0.25 → atenção
* > 0.25 → instável

* PSI do modelo: 0.0011
---

### 3️⃣ IV – Information Value

Avalia o **poder discriminante de cada variável**.

Referência:

| IV          | Interpretação                 |
| ----------- | ----------------------------- |
| < 0.02      | Inútil                        |
| 0.02 – 0.10 | Fraca                         |
| 0.10 – 0.30 | Boa                           |
| 0.30 – 0.50 | Muito forte                   |
| > 0.50      | Suspeita (possível vazamento) |

| Variável                     | IV       |
|-------------------------------|---------|
| loan_percent_income           | 0.872220 |
| loan_grade                    | 0.858246 |
| loan_int_rate                 | 0.614589 |
| person_income                 | 0.469187 |
| person_home_ownership         | 0.386428 |
| cb_person_default_on_file     | 0.159914 |
| loan_amnt                     | 0.089110 |
| loan_intent                   | 0.088281 |
| person_emp_length             | 0.058317 |
| person_age                    | 0.009781 |
| cb_person_cred_hist_length    | 0.004293 |

### 4️⃣ Monotonicidade

Avaliação se o risco (target médio) é monotônico ao longo dos bins.

* Obrigatório para scorecards
* Variáveis não monotônicas devem ser rebinadas

---

## ✅ Resultados Esperados

Um modelo bem-sucedido apresenta:

* AUC ≥ 0.80
* KS ≥ 0.40
* PSI < 0.10
* Variáveis com IV relevante
* Relações monotônicas estáveis

Ao analisar os indicadores acima. Podemos anlisar que o modelo apresenta:

* Excelente separação entre bons e maus pagadores.

* E que os resultados indicam que o modelo é adequado para apoiar decisões de crédito.
---

## 👤 Autor

Tony Eufrasio
Cientista de Dados / Analista de Risco de Crédito

---

