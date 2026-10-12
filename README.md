# Bank Marketing — Predição de conversão em campanhas de telemarketing

Projeto de **Machine Learning** sobre o dataset [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI): um banco português realizou campanhas para oferecer **depósito a prazo**. O objetivo é identificar clientes com maior probabilidade de aceitar a oferta, reduzindo custo de ligações e priorizando quem mais converte.

O fluxo completo está no notebook **`BancoMarketing.ipynb`** (EDA, pré-processamento, balanceamento, modelagem com **LightGBM** e exportação do modelo). A interface web em **Streamlit** (`app.py`) consome dados e artefatos pelo **Amazon S3**.

Projeto Integrador do Nível 1 da formação em Data Science da **Alura**, com foco em análise de dados, machine learning e computação em nuvem.

---

## Estrutura do repositório

| Arquivo / pasta | Descrição |
|-----------------|-----------|
| `BancoMarketing.ipynb` | Pipeline completo: EDA, feature engineering, balanceamento, LazyPredict, LightGBM, upload S3 |
| `app.py` | App Streamlit (EDA, simulador de campanha, inferência por cliente) |
| `requirements.txt` | Dependências Python |
| `novos_clientes_simulacao.csv` | Exemplos de clientes para testar inferência (mesmo schema do dataset) |
| `imagens_ilustrativas/` | Diagramas usados no notebook (balanceamento, LightGBM, threshold) |
| `dados/` | Dados disponíveis da página UCI |

---

## Pré-requisitos

| Ferramenta | Uso |
|------------|-----|
| **Python 3.10+** (recomendado) | Ambiente de execução |
| **AWS CLI** | Configurar credenciais e acesso ao S3 |
| **Conta AWS** | Bucket S3 com os dados e modelos |

### Instalar AWS CLI

- **macOS (Homebrew):** `brew install awscli`
- **Windows:** [Instalador oficial](https://aws.amazon.com/cli/) ou Chocolatey
- **Linux:** siga a [documentação da AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### Configurar credenciais AWS

1. No console AWS: **IAM** → usuário → **Security credentials** → **Create access key** (guarde Access Key e Secret com segurança).
2. No terminal:

```bash
aws configure
```

Informe:

- **AWS Access Key ID** e **Secret Access Key**
- **Default region name** (ex.: `us-east-1` ou a região do seu bucket)
- **Default output format** (pode deixar `json`)

O `boto3` usado no notebook e no `app.py` utiliza essas credenciais (ou variáveis de ambiente equivalentes).

### Bucket S3 esperado

O código assume o bucket **`bank-marketing-alura`** com esta estrutura:

| Caminho no bucket | Conteúdo |
|-------------------|----------|
| `raw/bank-full.csv` | Dataset bruto (`sep=';'`) |
| `modelos/modelo_lightgbm.pkl` | Modelo LightGBM treinado |
| `modelos/X_test_pre.pkl` | Features de teste pré-processadas |
| `modelos/y_test.pkl` | Rótulos do conjunto de teste |
| `modelos/ohe.pkl` | One-Hot Encoder (categóricas) |
| `modelos/oe.pkl` | Ordinal Encoder (`education`) |
| `modelos/scaler_age.pkl` | StandardScaler da idade |
| `modelos/scaler_balance.pkl` | StandardScaler do saldo (após log) |

Se usar outro nome de bucket ou prefixos, ajuste a constante `bucket` em `app.py` (e no notebook, onde houver upload/leitura do S3).

---

## Ambiente virtual (venv)

Na pasta do projeto:

```bash
cd /caminho/para/BankMarketing

python3 -m venv .venv
```

Ativar o ambiente:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Instalar dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependências principais (`requirements.txt`)

`boto3`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `streamlit`, `plotly`, `imbalanced-learn`, `lazypredict`, `matplotlib`, `seaborn`, `ipykernel`.

---

## Notebook `BancoMarketing.ipynb`

Resumo do que o notebook cobre:

1. **Setup e EDA** — Problema de negócio, AWS/S3, carregamento do CSV, exploração das variáveis e desbalanceamento de classes (~88% `no` / ~12% `yes`).
2. **Preparação para ML** — `train_test_split`, engenharia de features (One-Hot, Ordinal, StandardScaler, bins de `campaign`, tratamento de `balance`), balanceamento com **RandomOverSampler** + **RandomUnderSampler** (`imblearn`).
3. **Modelagem** — **LazyPredict** para varredura inicial; comparação manual (LightGBM, XGBoost, regressão logística); escolha do **LightGBM**; validação cruzada e matriz de confusão; persistência com `joblib` e upload para o S3.
4. **Streamlit** — Ideia e implementação do app: EDA + simulador de campanha com threshold e impacto em ligações/lucro + inferência individual.

Execute o notebook no Jupyter, VS Code ou Cursor com o kernel apontando para o mesmo ambiente onde instalou o `requirements.txt`.

> **Nota:** a variável `duration` (duração da ligação) é removida no pipeline de inferência do app porque só é conhecida **depois** do contato — usá-la na predição causaria *data leakage*.

---

## Aplicação Streamlit (`app.py`)

Com o venv ativado e o `aws configure` válido para uma conta que tenha **leitura** no bucket:

```bash
streamlit run app.py
```

Abra o endereço que o Streamlit mostrar no navegador (geralmente `http://localhost:8501`).

### Funcionalidades do app

| Página | O que faz |
|--------|-----------|
| **Análise exploratória** | Métricas resumidas, amostra dos dados, histograma de idade por conversão, taxa por profissão e série mensal |
| **Simulador de campanha** | Threshold sobre probabilidades, métricas de ligações/conversões, simulação financeira (receita/custo/lucro), curva de lucro por threshold, comparação com “ligar para todos”, matriz de confusão |
| **Inferência cliente** | Formulário com features do cliente; `predict_proba` + decisão “Ligar” / “Não ligar” conforme o threshold |
