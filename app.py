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
    pip install -r requirements.txt
    streamlit run app_streamlit.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

# Paleta "ficha de aluno / boletim escolar": tinta azul-marinho no papel
# creme claro, com um único destaque em âmbar (cor de marca-texto) para
# o resultado — em vez do terracota/creme genérico ou dos cards em série.
COR_TINTA = "#1B2A4A"        # azul-marinho (texto, cabeçalhos)
COR_PAPEL = "#F7F5EF"        # fundo claro tipo papel
COR_PAPEL_ESCURO = "#EFEBE1" # painéis/sidebar
COR_LINHA = "#D8D2C2"        # linhas de régua/divisores
COR_DESTAQUE = "#E2A33D"     # âmbar (marca-texto) — único acento forte
COR_SLATE = "#4C6485"        # azul acinzentado (texto secundário)

CORES_SITUACAO = {
    "Aprovado": "#3F7D5C",     # verde-musgo (aprovado no boletim)
    "Recuperacao": "#C98A1E",  # âmbar mais escuro (atenção)
    "Reprovado": "#A4463B",    # vermelho-tijolo (reprovado)
}

EMOJI_SITUACAO = {"Aprovado": "✅", "Recuperacao": "🟡", "Reprovado": "🔴"}

FONTE_SERIF = "'Source Serif 4', Georgia, serif"
FONTE_SANS = "'IBM Plex Sans', 'Segoe UI', sans-serif"


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
    """Ilustração da árvore treinada (mantida em matplotlib, tema combinando com a página)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(COR_PAPEL)
    ax.set_facecolor(COR_PAPEL)
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
        ax.set_title("Árvore de Decisão treinada", fontsize=13, fontweight="bold", color=COR_TINTA)
    else:
        ax.text(
            0.5, 0.5,
            "O modelo escolhido não é uma árvore de decisão única,\n"
            "então não há uma árvore individual para exibir.",
            ha="center", va="center", fontsize=11, wrap=True, color=COR_TINTA,
        )
        ax.axis("off")
    return fig


def gerar_grafico_matriz_confusao(y, previsoes, codificador):
    """Matriz de confusão interativa (Plotly): passe o mouse para ver os totais."""
    matriz = confusion_matrix(y, previsoes)
    classes = list(codificador.classes_)

    fig = px.imshow(
        matriz,
        x=classes,
        y=classes,
        color_continuous_scale=[[0, COR_PAPEL_ESCURO], [1, COR_TINTA]],
        labels=dict(x="Previsto", y="Real", color="Alunos"),
        text_auto=True,
    )
    fig.update_traces(
        hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Alunos: %{z}<extra></extra>",
        textfont=dict(family=FONTE_SANS, size=15, color=COR_TINTA),
    )
    fig.update_layout(
        title=dict(text="Matriz de Confusão (LOOCV)", font=dict(family=FONTE_SERIF, size=16, color=COR_TINTA)),
        font=dict(family=FONTE_SANS, color=COR_TINTA),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_showscale=False,
        height=360,
    )
    return fig


def gerar_grafico_comparacao(resultados: dict):
    """Comparação de modelos em barras interativas, destacando o vencedor em âmbar."""
    nomes = list(resultados.keys())
    acuracias = [resultados[n]["acuracia"] * 100 for n in nomes]
    melhor_idx = int(np.argmax(acuracias))
    cores = [COR_DESTAQUE if i == melhor_idx else COR_SLATE for i in range(len(nomes))]

    fig = go.Figure(
        go.Bar(
            x=nomes,
            y=acuracias,
            marker_color=cores,
            text=[f"{v:.0f}%" for v in acuracias],
            textposition="outside",
            hovertemplate="%{x}<br>Acurácia LOOCV: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Comparação entre modelos", font=dict(family=FONTE_SERIF, size=16, color=COR_TINTA)),
        yaxis=dict(title="Acurácia LOOCV (%)", range=[0, 105], gridcolor=COR_LINHA),
        xaxis=dict(showgrid=False),
        font=dict(family=FONTE_SANS, color=COR_TINTA),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        showlegend=False,
    )
    return fig


def gerar_grafico_probabilidades(prob_dict: dict):
    """Barras horizontais interativas com a probabilidade de cada situação."""
    classes = list(prob_dict.keys())
    valores = [v * 100 for v in prob_dict.values()]
    cores = [CORES_SITUACAO.get(c, COR_SLATE) for c in classes]

    fig = go.Figure(
        go.Bar(
            x=valores,
            y=classes,
            orientation="h",
            marker_color=cores,
            text=[f"{v:.1f}%" for v in valores],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Probabilidade (%)", range=[0, 105], gridcolor=COR_LINHA),
        yaxis=dict(showgrid=False, autorange="reversed"),
        font=dict(family=FONTE_SANS, color=COR_TINTA, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=190,
        showlegend=False,
    )
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


def dica_faltas(faltas: int) -> str:
    if faltas <= 5:
        return "🟢 frequência tranquila"
    if faltas <= 15:
        return "🟡 atenção à frequência"
    return "🔴 risco alto por faltas"


def dica_nota(nota: float) -> str:
    if nota >= 7:
        return "🟢 nota consolidada"
    if nota >= 5:
        return "🟡 nota na média"
    return "🔴 nota abaixo do esperado"


def dica_horas(horas: int) -> str:
    if horas >= 8:
        return "🟢 boa rotina de estudo"
    if horas >= 4:
        return "🟡 rotina moderada"
    return "🔴 pouco tempo de estudo"


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
    fig_matriz = gerar_grafico_matriz_confusao(
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
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, div {{
        font-family: {FONTE_SANS};
    }}
    h1, h2, h3, h4 {{
        font-family: {FONTE_SERIF};
        color: {COR_TINTA};
    }}

    .stApp {{
        background-color: {COR_PAPEL};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COR_PAPEL_ESCURO};
        border-right: 1px solid {COR_LINHA};
    }}
    div[data-testid="stMetric"] {{
        background-color: {COR_PAPEL};
        border: 1px solid {COR_LINHA};
        border-radius: 4px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {COR_TINTA};
        font-family: {FONTE_SERIF};
    }}

    .cabecalho-ficha {{
        border-top: 3px solid {COR_TINTA};
        border-bottom: 1px solid {COR_LINHA};
        padding: 18px 0 14px 0;
        margin-bottom: 22px;
    }}
    .cabecalho-ficha h1 {{
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: 0.2px;
    }}
    .cabecalho-ficha p {{
        margin: 4px 0 0 0;
        color: {COR_SLATE};
        font-size: 1rem;
    }}

    div[data-testid="stButton"] > button {{
        background-color: {COR_TINTA};
        color: {COR_PAPEL};
        border: none;
        border-radius: 4px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }}
    div[data-testid="stButton"] > button:hover {{
        background-color: {COR_SLATE};
        color: {COR_PAPEL};
    }}

    .dica-badge {{
        display: inline-block;
        font-size: 0.85rem;
        color: {COR_SLATE};
        margin-top: -6px;
        margin-bottom: 10px;
    }}

    .resultado-card {{
        border: 1px solid {COR_LINHA};
        border-left: 6px solid var(--cor-resultado, {COR_DESTAQUE});
        border-radius: 4px;
        padding: 22px 26px;
        background-color: white;
        animation: revelar 0.35s ease-out;
    }}
    @keyframes revelar {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .resultado-emoji {{ font-size: 2.4rem; line-height: 1; }}
    .resultado-classe {{
        font-family: {FONTE_SERIF};
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 4px;
    }}
    .resultado-confianca {{
        color: {COR_SLATE};
        font-size: 0.95rem;
        margin-top: 6px;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {COR_LINHA} !important;
        border-radius: 4px !important;
        background-color: white;
    }}

    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

dados = preparar_tudo()

# ---------------- Sidebar: ficha rápida do sistema ----------------
with st.sidebar:
    st.markdown("### 🎓 Ficha do sistema")
    st.metric("Modelo em uso", dados["nome_melhor_modelo"])
    st.metric(
        "Acurácia (LOOCV)",
        f"{dados['resultados_modelos'][dados['nome_melhor_modelo']]['acuracia'] * 100:.0f}%",
    )
    st.metric("Alunos na base", len(dados["df_alunos"]))
    st.markdown("---")
    st.markdown("**Legenda**")
    for situacao, cor in CORES_SITUACAO.items():
        st.markdown(
            f"<span style='color:{cor}; font-size:1.1rem;'>●</span> "
            f"{EMOJI_SITUACAO.get(situacao, '')} {situacao}",
            unsafe_allow_html=True,
        )
    if len(dados["df_alunos"]) < 30:
        st.markdown("---")
        st.caption(
            "⚠️ Base de dados pequena — resultados servem como demonstração "
            "da arquitetura, não como previsão estatisticamente validada."
        )

# ---------------- Cabeçalho principal ----------------
st.markdown(
    """
    <div class="cabecalho-ficha">
        <h1>🎓 Painel de Previsão de Situação Escolar</h1>
        <p>Estima se um aluno será Aprovado, entrará em Recuperação ou será Reprovado,
        a partir de horas de estudo, faltas e nota.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

aba_previsao, aba_modelo, aba_dados = st.tabs(
    ["🔮 Fazer Previsão", "📈 Sobre o Modelo", "📚 Dados de Treinamento"]
)

# ---------------- Aba 1: Previsão ----------------
with aba_previsao:
    col_entrada, col_resultado = st.columns([1, 1.2], gap="large")

    with col_entrada:
        st.markdown("#### 📋 Dados do aluno")

        entrada_horas = st.slider("Horas de estudo por semana", 0, 20, 5, step=1)
        st.markdown(f"<div class='dica-badge'>{dica_horas(entrada_horas)}</div>", unsafe_allow_html=True)

        entrada_faltas = st.slider("Número de faltas", 0, 30, 5, step=1)
        st.markdown(f"<div class='dica-badge'>{dica_faltas(entrada_faltas)}</div>", unsafe_allow_html=True)

        entrada_nota = st.slider("Nota obtida", 0.0, 10.0, 7.0, step=0.5)
        st.markdown(f"<div class='dica-badge'>{dica_nota(entrada_nota)}</div>", unsafe_allow_html=True)

        botao_prever = st.button("🔍 Prever situação", type="primary", use_container_width=True)

    with col_resultado:
        st.markdown("#### 🎯 Resultado")
        if botao_prever:
            classe_prevista, confianca, prob_dict, explicacao_ou_erros = prever_situacao(
                entrada_horas, entrada_faltas, entrada_nota,
                dados["modelo_final"], dados["escalador"], dados["codificador"],
            )

            if classe_prevista is None:
                for erro in explicacao_ou_erros:
                    st.warning(f"⚠️ {erro}")
            else:
                emoji = EMOJI_SITUACAO.get(classe_prevista, "")
                cor = CORES_SITUACAO.get(classe_prevista, COR_DESTAQUE)

                confianca_html = (
                    f"<div class='resultado-confianca'>Confiança do modelo: {confianca:.1f}%</div>"
                    if confianca is not None else ""
                )
                st.markdown(
                    f"""
                    <div class="resultado-card" style="--cor-resultado:{cor};">
                        <div class="resultado-emoji">{emoji}</div>
                        <div class="resultado-classe" style="color:{cor};">{classe_prevista}</div>
                        {confianca_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if classe_prevista == "Aprovado":
                    st.balloons()

                if confianca is not None:
                    st.progress(min(int(confianca), 100))

                with st.expander("📊 Probabilidades por situação", expanded=True):
                    if prob_dict is not None:
                        st.plotly_chart(
                            gerar_grafico_probabilidades(prob_dict),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    else:
                        st.markdown("Modelo não fornece probabilidades.")

                with st.expander("💡 Por que o modelo decidiu isso?", expanded=False):
                    st.markdown(explicacao_ou_erros)
        else:
            st.info("Ajuste os controles ao lado e clique em **Prever situação**.")

# ---------------- Aba 2: Sobre o modelo ----------------
with aba_modelo:
    tabela_modelos = "\n".join(
        f"| {nome} | {info['acuracia'] * 100:.0f}% |"
        for nome, info in dados["resultados_modelos"].items()
    )
    st.markdown(f"#### Modelo escolhido: **{dados['nome_melhor_modelo']}**")
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
        st.plotly_chart(dados["fig_comparacao"], use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(dados["fig_matriz"], use_container_width=True, config={"displayModeBar": False})
    st.pyplot(dados["fig_arvore"])

# ---------------- Aba 3: Dados utilizados ----------------
with aba_dados:
    st.markdown("#### Alunos usados para treinar o modelo")
    st.dataframe(dados["df_alunos"], use_container_width=True)
    st.markdown(
        f"Para adicionar mais alunos, edite ou crie um arquivo "
        f"`{CSV_PATH}` com as colunas `{', '.join(FEATURE_NAMES)}` "
        f"e `{TARGET_NAME}` — o sistema carrega esse arquivo "
        "automaticamente na próxima execução."
    )
