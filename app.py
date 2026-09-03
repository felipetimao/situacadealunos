"""
Sistema de Previsão de Situação Escolar
=========================================

Prevê se um aluno será Aprovado, ficará em Recuperação ou será Reprovado,
com base em horas de estudo, número de faltas e nota obtida.

Estrutura do arquivo (nessa ordem):
    1. Carregamento dos dados
    2. Preparação dos dados (encoding, escala)
    3. Comparação de modelos (avaliação)
    4. Treinamento do modelo final
    5. Funções de apoio (gráficos, previsão)
    6. Interface Gradio

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
"""

import io
import os
import base64

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import gradio as gr

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


def avaliar_modelos(x_escalado, y, n_amostras: int):
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

def figura_para_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    codigo = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{codigo}"


def gerar_grafico_arvore(modelo, codificador) -> str:
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
    return figura_para_base64(fig)


def gerar_matriz_confusao(y, previsoes, codificador) -> str:
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
    return figura_para_base64(fig)


def gerar_grafico_comparacao(resultados: dict) -> str:
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
    return figura_para_base64(fig)


def explicar_previsao(modelo, codificador, x_linha_escalada) -> str:
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
    return (
        f"**Total de registros:** {len(df)}\n\n{linhas}{aviso}"
    )


# ---------------------------------------------------------------------------
# Preparação executada uma única vez, na inicialização do app
# ---------------------------------------------------------------------------

df_alunos = carregar_dados()
x_bruto, x_escalado, y_codificado, codificador, escalador = preparar_dados(df_alunos)
resultados_modelos, nome_melhor_modelo = avaliar_modelos(x_escalado, y_codificado, len(df_alunos))
modelo_final = treinar_modelo_final(nome_melhor_modelo, x_escalado, y_codificado)

GRAFICO_ARVORE = gerar_grafico_arvore(
    montar_candidatos()["Árvore de Decisão"].fit(x_escalado, y_codificado), codificador
)
GRAFICO_MATRIZ = gerar_matriz_confusao(
    y_codificado, resultados_modelos[nome_melhor_modelo]["previsoes"], codificador
)
GRAFICO_COMPARACAO = gerar_grafico_comparacao(resultados_modelos)
RESUMO_DADOS = resumo_dataset(df_alunos)

TEXTO_SOBRE_MODELO = f"""
### Modelo escolhido: **{nome_melhor_modelo}**

| Modelo | Acurácia (LOOCV) |
|---|---|
""" + "\n".join(
    f"| {nome} | {info['acuracia'] * 100:.0f}% |"
    for nome, info in resultados_modelos.items()
) + f"""

O modelo foi escolhido comparando **Árvore de Decisão**, **Random Forest** e
**Regressão Logística** usando **Leave-One-Out Cross-Validation (LOOCV)** —
a técnica correta para datasets muito pequenos, já que um único
train/test split com poucos dados dá uma métrica pouco confiável.

Depois da comparação, o modelo vencedor é retreinado com **100% dos dados**
disponíveis para ser usado nas previsões.

{RESUMO_DADOS}
"""


# ---------------------------------------------------------------------------
# 6. Função de previsão usada pela interface
# ---------------------------------------------------------------------------

def prever_situacao(horas_estudo, faltas, nota):
    if horas_estudo is None or faltas is None or nota is None:
        return (
            "⚠️ Preencha todos os campos.", "", ""
        )

    erros = []
    if horas_estudo < 0:
        erros.append("Horas de estudo não pode ser negativo.")
    if faltas < 0:
        erros.append("Faltas não pode ser negativo.")
    if not (0 <= nota <= 10):
        erros.append("Nota deve estar entre 0 e 10.")

    if erros:
        return ("⚠️ " + " ".join(erros), "", "")

    entrada = pd.DataFrame([[horas_estudo, faltas, nota]], columns=FEATURE_NAMES)
    entrada_escalada = escalador.transform(entrada)

    classe_prevista_idx = modelo_final.predict(entrada_escalada)[0]
    classe_prevista = codificador.inverse_transform([classe_prevista_idx])[0]

    if hasattr(modelo_final, "predict_proba"):
        probabilidades = modelo_final.predict_proba(entrada_escalada)[0]
        prob_texto = "\n".join(
            f"- **{classe}**: {p * 100:.1f}%"
            for classe, p in sorted(
                zip(codificador.classes_, probabilidades),
                key=lambda item: item[1], reverse=True
            )
        )
        confianca = max(probabilidades) * 100
    else:
        prob_texto = "Modelo não fornece probabilidades."
        confianca = None

    explicacao = explicar_previsao(modelo_final, codificador, entrada_escalada)

    emoji = {"Aprovado": "✅", "Recuperacao": "🟡", "Reprovado": "🔴"}.get(classe_prevista, "")
    cor = CORES_SITUACAO.get(classe_prevista, "#6366f1")
    resultado_html = f"""
    <div style="background:{cor}22; border:2px solid {cor}; border-radius:16px;
                padding:24px; text-align:center;">
        <div style="font-size:42px;">{emoji}</div>
        <div style="font-size:28px; font-weight:800; color:{cor}; margin-top:4px;">
            {classe_prevista}
        </div>
        {"<div style='font-size:14px; color:#555; margin-top:6px;'>Confiança do modelo: " + f"{confianca:.1f}%" + "</div>" if confianca is not None else ""}
    </div>
    """

    return resultado_html, prob_texto, explicacao


# ---------------------------------------------------------------------------
# Interface Gradio — visual estilo dashboard
# ---------------------------------------------------------------------------

CSS_PERSONALIZADO = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#titulo-principal {text-align: center; margin-bottom: 0;}
#subtitulo {text-align: center; color: #6b7280; margin-top: 0;}
.card {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    padding: 16px !important;
}
"""

with gr.Blocks(css=CSS_PERSONALIZADO, theme=gr.themes.Soft(primary_hue="indigo")) as interface:
    gr.Markdown("# 🎓 Painel de Previsão de Situação Escolar", elem_id="titulo-principal")
    gr.Markdown(
        "Modelo de Machine Learning que estima se um aluno será "
        "**Aprovado**, entrará em **Recuperação** ou será **Reprovado**.",
        elem_id="subtitulo",
    )

    with gr.Tabs():
        # ---------------- Aba 1: Previsão ----------------
        with gr.Tab("🔮 Fazer Previsão"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="card"):
                    gr.Markdown("### 📋 Dados do aluno")
                    entrada_horas = gr.Slider(
                        0, 20, value=5, step=1, label="Horas de estudo por semana"
                    )
                    entrada_faltas = gr.Slider(
                        0, 30, value=5, step=1, label="Número de faltas"
                    )
                    entrada_nota = gr.Slider(
                        0, 10, value=7, step=0.5, label="Nota obtida"
                    )
                    botao_prever = gr.Button("🔍 Prever situação", variant="primary")

                with gr.Column(scale=1, elem_classes="card"):
                    gr.Markdown("### 🎯 Resultado")
                    saida_resultado = gr.HTML()
                    with gr.Accordion("📊 Probabilidades por situação", open=True):
                        saida_probabilidades = gr.Markdown()
                    with gr.Accordion("💡 Por que o modelo decidiu isso?", open=False):
                        saida_explicacao = gr.Markdown()

            botao_prever.click(
                fn=prever_situacao,
                inputs=[entrada_horas, entrada_faltas, entrada_nota],
                outputs=[saida_resultado, saida_probabilidades, saida_explicacao],
            )

        # ---------------- Aba 2: Sobre o modelo ----------------
        with gr.Tab("📈 Sobre o Modelo"):
            gr.Markdown(TEXTO_SOBRE_MODELO)
            with gr.Row():
                gr.Image(GRAFICO_COMPARACAO, label="Comparação de modelos", show_label=True)
                gr.Image(GRAFICO_MATRIZ, label="Matriz de confusão", show_label=True)
            gr.Image(GRAFICO_ARVORE, label="Árvore de decisão (referência visual)")

        # ---------------- Aba 3: Dados utilizados ----------------
        with gr.Tab("📚 Dados de Treinamento"):
            gr.Markdown("### Alunos usados para treinar o modelo")
            gr.Dataframe(df_alunos, interactive=False)
            gr.Markdown(
                f"Para adicionar mais alunos, edite ou crie um arquivo "
                f"`{CSV_PATH}` com as colunas `{', '.join(FEATURE_NAMES)}` "
                f"e `{TARGET_NAME}` — o sistema carrega esse arquivo "
                "automaticamente na próxima execução."
            )


if __name__ == "__main__":
    interface.launch()
