# 🎓 Painel de Previsão de Situação Escolar

Aplicação de Machine Learning com interface interativa (Gradio) que prevê se um aluno será **Aprovado**, ficará em **Recuperação** ou será **Reprovado**, com base em:

- Horas de estudo por semana
- Número de faltas
- Nota obtida

O projeto compara automaticamente diferentes modelos de classificação (Árvore de Decisão, Random Forest e Regressão Logística) e usa o de melhor desempenho para gerar as previsões.

---

## ⚠️ Aviso importante sobre os dados

O dataset incluído no projeto tem **apenas 5 alunos** (dados de exemplo). Isso é insuficiente para um modelo estatisticamente confiável — é fácil o modelo "decorar" os poucos exemplos existentes em vez de aprender um padrão real.

Por isso, o código usa **Leave-One-Out Cross-Validation (LOOCV)**, a técnica mais honesta para avaliar modelos com poucos dados, e deixa esse alerta visível na própria interface (aba "Sobre o Modelo").

**Este projeto deve ser tratado como uma demonstração de arquitetura de ML, não como uma ferramenta de decisão real**, até que seja treinado com uma base de dados maior (idealmente algumas dezenas de registros por classe).

---

## ✨ Funcionalidades

- Interface moderna em abas: **Fazer Previsão**, **Sobre o Modelo** e **Dados de Treinamento**
- Previsão com probabilidade/confiança por classe
- Explicação simples de quais variáveis mais influenciaram a decisão
- Comparação visual entre modelos (gráfico de acurácia)
- Matriz de confusão e visualização da árvore de decisão
- Validação de entradas inválidas (valores negativos, nota fora de 0–10)
- Estrutura pronta para receber uma base de dados maior (arquivo CSV externo)

---

## 📦 Instalação

```bash
pip install pandas scikit-learn matplotlib gradio
```

## ▶️ Como executar localmente

```bash
python app.py
```

A interface abre em `http://localhost:7860`.

## ▶️ Como executar no Google Colab

```python
!pip install gradio -q
```

E, no final do script, use:

```python
interface.launch(share=True, debug=True)
```

O parâmetro `share=True` gera um link público temporário (necessário porque o Colab não expõe `localhost` diretamente) e `debug=True` ajuda a visualizar erros no próprio notebook.

---

## 📁 Adicionando mais dados

Para treinar o modelo com uma base maior, crie um arquivo `alunos.csv` na mesma pasta do `app.py`, com as colunas:

```
Horas_de_estudo,Faltas,Nota,Situacao
```

O script carrega esse arquivo automaticamente na próxima execução — nenhuma alteração de código é necessária.

---

## 🧠 Modelos comparados

| Modelo | Método de avaliação |
|---|---|
| Árvore de Decisão | Leave-One-Out Cross-Validation |
| Random Forest | Leave-One-Out Cross-Validation |
| Regressão Logística | Leave-One-Out Cross-Validation |

O modelo com melhor acurácia é retreinado com 100% dos dados disponíveis para uso na interface.

---

## 🗂️ Estrutura do código

```
app.py
├── Carregamento dos dados      (carregar_dados)
├── Preparação dos dados        (preparar_dados)
├── Comparação de modelos       (avaliar_modelos)
├── Treinamento do modelo final (treinar_modelo_final)
├── Funções de apoio            (gráficos, explicação, resumo)
└── Interface Gradio            (gr.Blocks)
```

## 📄 Licença

Defina aqui a licença do seu repositório (ex.: MIT).
