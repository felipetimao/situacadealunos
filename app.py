"""
Sistema de Previsão de Situação Escolar (versão Streamlit)
============================================================

Prevê se um aluno será Aprovado, ficará em Recuperação ou será Reprovado,
com base em horas de estudo, número de faltas e nota obtida.

Estrutura do arquivo (nessa ordem):
    1. Carregamento dos dados
    2. Preparação dos dados (encoding, escala)
    3. Comparação de modelos (avaliação)
    4. Treinamento do modelo final
    5. Funções de apoio (gráficos, previsão)
    6. Interface Streamlit

⚠️ AVISO IMPORTANTE SOBRE OS DADOS
-----------------------------------
O conjunto de dados atual tem apenas 5 alunos. Isso é MUITO pouco para
treinar um modelo de Machine Learning confiável. Qualquer "acurácia" que
o modelo apresente com um dataset desse tamanho tem valor estatístico
quase nulo — é fácil o modelo "decorar" os poucos exemplos existentes
(overfitting) em vez de aprender um padrão real.

Por isso:
    - Trocamos o train_test_split (que com 5 linhas deixaria o teste com
      1 único aluno, tornando a métrica praticamente aleatória) por
      Leave-One-Out Cross-Validation (LOOCV), a técnica mais adequada
      para bases de dados muito pequenas: cada aluno é usado como teste
      exatamente uma vez, e todos os outros são usados no treino.
    - O código foi estruturado para carregar os dados de um arquivo CSV
      externo (alunos.csv), caso ele exista. Assim, quando você tiver
      mais dados reais de alunos, basta criar/atualizar esse CSV — nada
      no código precisa mudar.
    - O painel "Sobre o Modelo" na interface deixa esse alerta visível
      para quem for usar o sistema, para que ninguém confie demais numa
      previsão baseada em tão poucos exemplos.

Recomendação: para um modelo realmente confiável, o ideal é ter pelo
menos algumas dezenas (idealmente centenas) de registros de alunos.

Para rodar:
    streamlit run app_streamlit.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["Horas_de_estudo", "Faltas", "Nota"]
TARGET_NAME = "Situacao"
CSV_PATH = "alunos.csv"          # coloque aqui um arquivo maior no futuro
RANDOM_STATE = 42

CORES_SITUACAO = {
    "Aprovado": "#22c55e",
    "Recuperacao": "#f59e0b",
    "Reprovado": "#ef4444",
}


# ---------------------------------------------------------------------------
# 1. Carregamento dos dados
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """
    Carrega o dataset de alunos.

    Se existir um arquivo `alunos.csv` na pasta do projeto, ele é usado
    (facilitando a expansão futura da base). Caso contrário, usa-se o
    conjunto de dados original de exemplo (apenas 5 alunos).
    """
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df

    dados = {
        "Horas_de_estudo": [10, 2, 5, 8, 1],
        "Faltas": [2, 15, 6, 1, 20],
        "Nota": [8.5, 3.0, 6.5, 9.0, 2.5],
        "Situacao": ["Aprovado", "Reprovado", "Recuperacao", "Aprovado", "Reprovado"],
    }
    return pd.DataFrame(dados)


# ---------------------------------------------------------------------------
# 2. Preparação dos dados
# ---------------------------------------------------------------------------

def preparar_dados(df: pd.DataFrame):
    """
    Separa features e alvo, codifica as classes e padroniza as
    variáveis numéricas (importante para a Regressão Logística; não
    prejudica as árvores/floresta).
    """
    x = df[FEATURE_NAMES].copy()
    y_texto = df[TARGET_NAME].copy()

    codificador = LabelEncoder()
    y = codificador.fit_transform(y_texto)

    escalador = StandardScaler()
    x_escalado = escalador.fit_transform(x)

    return x, x_escalado, y, codificador, escalador


# ---------------------------------------------------------------------------
# 3. Comparação de modelos (avaliação honesta com poucos dados)
# ---------------------------------------------------------------------------

def montar_candidatos():
    """Modelos candidatos, todos com random_state fixo (reprodutibilidade)."""
    return {
        "Árvore de Decisão": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=3, min_samples_leaf=1
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=3
        ),
        "Regressão Logística": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000
        ),
    }


def avaliar_modelos(x_escalado, y):
    """
    Avalia cada modelo candidato usando Leave-One-Out Cross-Validation.

    Com bases de dados muito pequenas (poucas dezenas de linhas ou
    menos), LOOCV aproveita cada exemplo tanto para treino quanto para
    teste, dando uma estimativa muito mais estável do que um único
    train_test_split, que com poucos dados dependeria de "sorte" na
    divisão.
    """
    loo = LeaveOneOut()
    resultados = {}

    for nome, modelo in montar_candidatos().items():
        # Com bases muito pequenas, alguma classe pode ter só 1 exemplo;
        # nesse caso o LOOCV ainda funciona (o exemplo problemático só
        # não terá "colegas" da mesma classe no treino em uma rodada).
        previsoes = cross_val_predict(modelo, x_escalado, y, cv=loo)
        acuracia = accuracy_score(y, previsoes)
        resultados[nome] = {
            "acuracia": acuracia,
            "previsoes": previsoes,
        }

    melhor_nome = max(resultados, key=lambda k: resultados[k]["acuracia"])
    return resultados, melhor_nome


# ---------------------------------------------------------------------------
# 4. Treinamento do modelo final
# ---------------------------------------------------------------------------

def treinar_modelo_final(nome_modelo: str, x_escalado, y):
    """
    Treina o modelo escolhido com TODOS os dados disponíveis.

    Com apenas alguns exemplos, reservar uma fatia só para teste faria
    o modelo final "perder" informação valiosa. Por isso, depois de
    avaliar os modelos de forma honesta (LOOCV, acima), o modelo
    campeão é retreinado com 100% dos dados para uso em produção.
    """
    modelo = montar_candidatos()[nome_modelo]
    modelo.fit(x_escalado, y)
    return modelo


# ---------------------------------------------------------------------------
# 5. Funções de apoio (gráficos e explicações)
# ---------------------------------------------------------------------------

def gerar_grafico_arvore(modelo, codificador):
    fig, ax = plt.subplots(figsize=(9, 6))
    if isinstance(modelo, DecisionTreeClassifier):
        plot_tree(
            modelo,
            feature_names=FEATURE_NAMES,
            class_names=[str(c) for c in codificador.classes_],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
        )
        ax.set_title("Árvore de Decisão treinada", fontsize=13, fontweight="bold")
    else:
        ax.text(
            0.5, 0.5,
            "O modelo escolhido não é uma árvore de decisão única,\n"
            "então não há uma árvore individual para exibir.",
            ha="center", va="center", fontsize=11, wrap=True,
        )
        ax.axis("off")
    return fig


def gerar_matriz_confusao(y, previsoes, codificador):
    matriz = confusion_matrix(y, previsoes)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matriz, cmap="Blues")
    classes = codificador.classes_
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (LOOCV)", fontsize=12, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matriz[i, j], ha="center", va="center",
                     color="white" if matriz[i, j] > matriz.max() / 2 else "black",
                     fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def gerar_grafico_comparacao(resultados: dict):
    nomes = list(resultados.keys())
    acuracias = [resultados[n]["acuracia"] * 100 for n in nomes]
    cores = ["#6366f1", "#06b6d4", "#f97316"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    barras = ax.bar(nomes, acuracias, color=cores[: len(nomes)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acurácia LOOCV (%)")
    ax.set_title("Comparação entre modelos", fontsize=12, fontweight="bold")
    for barra, valor in zip(barras, acuracias):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 2,
                 f"{valor:.0f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=10)
    return fig


def explicar_previsao(modelo, x_linha_escalada) -> str:
    """Gera uma explicação textual simples com base na importância das features."""
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_).mean(axis=0)
    else:
        return "Este modelo não fornece uma explicação detalhada da decisão."

    ordem = np.argsort(importancias)[::-1]
    principais = [FEATURE_NAMES[i] for i in ordem[:2]]
    return (
        f"A previsão foi mais influenciada por **{principais[0]}** "
        f"e **{principais[1]}**, que são os fatores de maior peso "
        f"aprendidos pelo modelo."
    )


def resumo_dataset(df: pd.DataFrame) -> str:
    contagem = df[TARGET_NAME].value_counts()
    linhas = "\n".join(f"- **{classe}**: {qtd} aluno(s)" for classe, qtd in contagem.items())
    aviso = ""
    if len(df) < 30:
        aviso = (
            "\n\n⚠️ **Atenção:** o dataset tem apenas "
            f"**{len(df)} registros**. Isso é insuficiente para um modelo "
            "estatisticamente confiável — os resultados abaixo servem como "
            "demonstração da arquitetura do projeto, não como uma previsão "
            "validada de verdade. Para confiabilidade real, recomenda-se "
            "coletar pelo menos algumas dezenas de exemplos por classe."
        )
    return f"**Total de registros:** {len(df)}\n\n{linhas}{aviso}"


# ---------------------------------------------------------------------------
# Preparação executada uma única vez (cacheada) na inicialização do app
# ---------------------------------------------------------------------------

@st.cache_resource
def preparar_tudo():
    df_alunos = carregar_dados()
    x_bruto, x_escalado, y_codificado, codificador, escalador = preparar_dados(df_alunos)
    resultados_modelos, nome_melhor_modelo = avaliar_modelos(x_escalado, y_codificado)
    modelo_final = treinar_modelo_final(nome_melhor_modelo, x_escalado, y_codificado)

    fig_arvore = gerar_grafico_arvore(
        montar_candidatos()["Árvore de Decisão"].fit(x_escalado, y_codificado), codificador
    )
    fig_matriz = gerar_matriz_confusao(
        y_codificado, resultados_modelos[nome_melhor_modelo]["previsoes"], codificador
    )
    fig_comparacao = gerar_grafico_comparacao(resultados_modelos)
    resumo_dados = resumo_dataset(df_alunos)

    return {
        "df_alunos": df_alunos,
        "x_escalado": x_escalado,
        "y_codificado": y_codificado,
        "codificador": codificador,
        "escalador": escalador,
        "resultados_modelos": resultados_modelos,
        "nome_melhor_modelo": nome_melhor_modelo,
        "modelo_final": modelo_final,
        "fig_arvore": fig_arvore,
        "fig_matriz": fig_matriz,
        "fig_comparacao": fig_comparacao,
        "resumo_dados": resumo_dados,
    }


# ---------------------------------------------------------------------------
# 6. Função de previsão usada pela interface
# ---------------------------------------------------------------------------

def prever_situacao(horas_estudo, faltas, nota, modelo_final, escalador, codificador):
    erros = []
    if horas_estudo < 0:
        erros.append("Horas de estudo não pode ser negativo.")
    if faltas < 0:
        erros.append("Faltas não pode ser negativo.")
    if not (0 <= nota <= 10):
        erros.append("Nota deve estar entre 0 e 10.")

    if erros:
        return None, None, None, erros

    entrada = pd.DataFrame([[horas_estudo, faltas, nota]], columns=FEATURE_NAMES)
    entrada_escalada = escalador.transform(entrada)

    classe_prevista_idx = modelo_final.predict(entrada_escalada)[0]
    classe_prevista = codificador.inverse_transform([classe_prevista_idx])[0]

    confianca = None
    prob_dict = None
    if hasattr(modelo_final, "predict_proba"):
        probabilidades = modelo_final.predict_proba(entrada_escalada)[0]
        prob_dict = dict(
            sorted(
                zip(codificador.classes_, probabilidades),
                key=lambda item: item[1], reverse=True
            )
        )
        confianca = max(probabilidades) * 100

    explicacao = explicar_previsao(modelo_final, entrada_escalada)

    return classe_prevista, confianca, prob_dict, explicacao


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Previsão de Situação Escolar",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; margin: auto;}
    #titulo-principal {text-align: center;}
    .card {
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 id='titulo-principal'>🎓 Painel de Previsão de Situação Escolar</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>Modelo de Machine Learning que estima se um aluno será "
    "<b>Aprovado</b>, entrará em <b>Recuperação</b> ou será <b>Reprovado</b>.</p>",
    unsafe_allow_html=True,
)

dados = preparar_tudo()

aba_previsao, aba_modelo, aba_dados = st.tabs(
    ["🔮 Fazer Previsão", "📈 Sobre o Modelo", "📚 Dados de Treinamento"]
)

# ---------------- Aba 1: Previsão ----------------
with aba_previsao:
    col_entrada, col_resultado = st.columns(2)

    with col_entrada:
        st.markdown("### 📋 Dados do aluno")
        entrada_horas = st.slider("Horas de estudo por semana", 0, 20, 5, step=1)
        entrada_faltas = st.slider("Número de faltas", 0, 30, 5, step=1)
        entrada_nota = st.slider("Nota obtida", 0.0, 10.0, 7.0, step=0.5)
        botao_prever = st.button("🔍 Prever situação", type="primary")

    with col_resultado:
        st.markdown("### 🎯 Resultado")
        if botao_prever:
            classe_prevista, confianca, prob_dict, explicacao_ou_erros = prever_situacao(
                entrada_horas, entrada_faltas, entrada_nota,
                dados["modelo_final"], dados["escalador"], dados["codificador"],
            )

            if classe_prevista is None:
                for erro in explicacao_ou_erros:
                    st.warning(f"⚠️ {erro}")
            else:
                emoji = {"Aprovado": "✅", "Recuperacao": "🟡", "Reprovado": "🔴"}.get(classe_prevista, "")
                cor = CORES_SITUACAO.get(classe_prevista, "#6366f1")

                confianca_html = (
                    f"<div style='font-size:14px; color:#555; margin-top:6px;'>"
                    f"Confiança do modelo: {confianca:.1f}%</div>"
                    if confianca is not None else ""
                )
                st.markdown(
                    f"""
                    <div style="background:{cor}22; border:2px solid {cor}; border-radius:16px;
                                padding:24px; text-align:center;">
                        <div style="font-size:42px;">{emoji}</div>
                        <div style="font-size:28px; font-weight:800; color:{cor}; margin-top:4px;">
                            {classe_prevista}
                        </div>
                        {confianca_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📊 Probabilidades por situação", expanded=True):
                    if prob_dict is not None:
                        for classe, p in prob_dict.items():
                            st.markdown(f"- **{classe}**: {p * 100:.1f}%")
                    else:
                        st.markdown("Modelo não fornece probabilidades.")

                with st.expander("💡 Por que o modelo decidiu isso?", expanded=False):
                    st.markdown(explicacao_ou_erros)
        else:
            st.info("Preencha os dados e clique em **Prever situação**.")

# ---------------- Aba 2: Sobre o modelo ----------------
with aba_modelo:
    tabela_modelos = "\n".join(
        f"| {nome} | {info['acuracia'] * 100:.0f}% |"
        for nome, info in dados["resultados_modelos"].items()
    )
    st.markdown(f"### Modelo escolhido: **{dados['nome_melhor_modelo']}**")
    st.markdown(
        "| Modelo | Acurácia (LOOCV) |\n|---|---|\n" + tabela_modelos
    )
    st.markdown(
        """
        O modelo foi escolhido comparando **Árvore de Decisão**, **Random Forest** e
        **Regressão Logística** usando **Leave-One-Out Cross-Validation (LOOCV)** —
        a técnica correta para datasets muito pequenos, já que um único
        train/test split com poucos dados dá uma métrica pouco confiável.

        Depois da comparação, o modelo vencedor é retreinado com **100% dos dados**
        disponíveis para ser usado nas previsões.
        """
    )
    st.markdown(dados["resumo_dados"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(dados["fig_comparacao"])
    with col_b:
        st.pyplot(dados["fig_matriz"])
    st.pyplot(dados["fig_arvore"])

# ---------------- Aba 3: Dados utilizados ----------------
with aba_dados:
    st.markdown("### Alunos usados para treinar o modelo")
    st.dataframe(dados["df_alunos"], use_container_width=True)
    st.markdown(
        f"Para adicionar mais alunos, edite ou crie um arquivo "
        f"`{CSV_PATH}` com as colunas `{', '.join(FEATURE_NAMES)}` "
        f"e `{TARGET_NAME}` — o sistema carrega esse arquivo "
        "automaticamente na próxima execução."
    )
