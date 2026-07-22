# Aprendizado de Máquina — UFRB

Repositório com os trabalhos desenvolvidos nas disciplinas **Aprendizado de Máquina** (Mestrado e Tópicos Especiais em Inteligência Artificial) e **Aprendizado de Máquina** (Graduação), do curso de **Bacharelado em Engenharia de Computação** e **Pós-graduação em Engenharia Elétrica e Computação** — CETEC / UFRB.

**Alunos:** Jeovane dos Santos, Quele Andrade, Luana Lima, Renatta Andrade

---

## Sumário

| # | Trabalho | Técnica | Arquivos |
|---|----------|---------|----------|
| 1 | [Detecção de Desmatamento](#1-detecção-de-desmatamento-com-redes-neurais-convolucionais) | CNN (ResNet-50 + Transfer Learning) | [`CNN_Deforestation_v4.ipynb`](CNN_Deforestation_v4.ipynb), [`Detecção de Desmatamento.pdf`](Detecção%20de%20Desmatamento.pdf) |
| 2 | [Previsão de Preços de Gasolina](#2-previsão-de-preços-de-gasolina-com-redes-neurais-mlp) | MLP (séries temporais) | [`regressao_gasolina.ipynb`](regressao_gasolina.ipynb), [`Previsão de Preços de gasolina.pdf`](Previsão%20de%20Preços%20de%20gasolina.pdf) |
| 3 | [Classificação de Doenças em Plantas](#3-classificação-de-doenças-em-plantas-com-transfer-learning) | CNN (EfficientNetB0 + Transfer Learning) | [`plantdoc/plantdoc_classificacaoVersion2.ipynb`](plantdoc/plantdoc_classificacaoVersion2.ipynb), [`Classificação de Doenças em plantas.pdf`](Classificação%20de%20Doenças%20em%20plantas.pdf) |
| 4 | [Clusterização de Doenças em Plantas](#4-clusterização-de-doenças-em-plantas-plantdoc) | Não supervisionado (EfficientNetB0 + PCA + K-Means) | [`plantdoc/plantdoc_clusterizacao_correto.ipynb`](plantdoc/plantdoc_clusterizacao_correto.ipynb) |
| 5 | [LLM — Comparação de Desempenho](#5-llm--comparação-de-desempenho) | LLMs locais + RAG | [`LLM/`](LLM/) |

---

## 1. Detecção de Desmatamento com Redes Neurais Convolucionais

Classificação binária de imagens de satélite (**desmatado** vs. **preservado**) usando **ResNet-50** com Transfer Learning em duas etapas (cabeça + fine-tuning), Data Augmentation e a função de perda `BinaryFocalCrossentropy` para lidar com classes desbalanceadas e escassez de dados.

- **Dataset:** Kaggle *Deforestation / Non-Deforestation Area Analysis* — 130 imagens (77 desmatamento / 53 não desmatamento), split 80/20 (103 treino / 27 teste).
- **Modelo:** ResNet-50 pré-treinada (ImageNet). Estágio 1 — backbone congelado, treina só a cabeça (LR=1e-3, val_acc 76,96%... base). Estágio 2 — fine-tuning dos blocos conv4+conv5 (LR=1e-6).
- **Imagem:** 224×224 px · Batch size 16 · Otimizador Adam.

### Resultados (conjunto de teste — 22/27 imagens avaliadas)

| Métrica | Desmatamento | Não desmatamento |
|---|---|---|
| Precision | 0,89 | 1,00 |
| Recall | 1,00 | 0,82 |
| F1-Score | 0,94 | 0,90 |

- **Acurácia:** 93%
- **AUC-ROC:** 0,9773
- **F1 macro:** 0,92

**Matriz de confusão (threshold = 0,33):**

|  | Predito: deforestation | Predito: nodeforestation |
|---|---|---|
| **Real: deforestation** | 16 | 0 |
| **Real: nodeforestation** | 2 | 9 |

### Comparação com a literatura

| Trabalho | Método | Métrica | Valor |
|---|---|---|---|
| Dastour & Hassan (2023) | ResNet-50 + TL | F1-score | 0,967 |
| Al-Dabbagh et al. (2024) | U-Net + ResNet-50 | Acurácia | 99,91% |
| Bayrak & Demiray (2024) | CNNs pré-treinadas | Δ Acurácia | −10% |
| **Este trabalho** | **ResNet-50 + TL** | **F1 / AUC-ROC** | **0,92 / 0,9773** |

**Conclusão:** mesmo com apenas 130 imagens, o pipeline de duas etapas (Transfer Learning + fine-tuning) combinado com Data Augmentation agressiva e Focal Loss viabilizou resultados consistentes com a literatura. Como trabalho futuro, aponta-se a necessidade de segmentação semântica para maior precisão operacional.

---

## 2. Previsão de Preços de Gasolina com Redes Neurais MLP

Aplicação de redes **MLP (Multilayer Perceptron)** para regressão em séries temporais do preço médio de revenda da gasolina comum no Brasil, comparando três estratégias de representação temporal.

- **Dataset:** Kaggle *combustiveis-brasil* — preço médio mensal de revenda, múltiplos anos, split 90/10.
- **Arquitetura:** `Dense(36) → Dense(18) → Dense(8) → Dense(1)`, ativação ReLU, saída linear, loss MSE, otimizador Adam, 200 épocas, `MinMaxScaler`.
- **Estratégias testadas:** Discretização numérica (ano/mês como inteiros), Codificação trigonométrica (seno de ano/mês) e Janelamento (*windowing*) com n=2, 4 e 10 passos anteriores.

### Resultados — comparação entre abordagens (conjunto de teste)

| Abordagem | Entradas | MSE | MAE | R² |
|---|---|---|---|---|
| Discretização | ano, mês, pₜ | 13,18% | 29,48% | 83,06% |
| Trigonométrica | cos(mês), sin(mês), pₜ | 13,44% | 25,17% | 82,72% |
| Janelamento n=2 | pₜ₋₁, pₜ | 10,53% | 24,27% | 86,46% |
| Janelamento n=4 | pₜ₋₃…pₜ | 9,85% | 24,27% | 89,09% |
| **Janelamento n=10** | pₜ₋₉…pₜ | **8,86%** | **24,43%** | **92,14%** |

O janelamento com **n=10** apresentou o melhor R² (92,14%), indicando maior capacidade de capturar a tendência da série apenas com o histórico de preços como entrada — resultado competitivo mesmo frente a arquiteturas mais complexas (LSTM, NARNN, Random Forest) usadas em trabalhos relacionados.

### Comparação com a literatura

| Trabalho | Modelo | Domínio | R² |
|---|---|---|---|
| Alwadi (2025) | Random Forest | Combustível (Jordânia) | 98,60% |
| Jin & Xu (2024) | NARNN | Cru/gás (global, diário) | >91,00% |
| **Este trabalho** | MLP — janela n=10 | Gasolina (Brasil, mensal) | **92,14%** |
| Este trabalho | MLP — janela n=4 | Gasolina (Brasil, mensal) | 89,09% |
| Este trabalho | MLP — janela n=2 | Gasolina (Brasil, mensal) | 86,46% |

**Limitações e trabalhos futuros:** o MLP não possui memória sequencial explícita; próximos passos incluem testar LSTM, ampliar a janela para n=12 (sazonalidade anual) e incorporar variáveis exógenas (câmbio, preço do barril de petróleo).

---

## 3. Classificação de Doenças em Plantas com Transfer Learning

Classificação de 30 categorias de doenças foliares (13 espécies de plantas) usando **EfficientNetB0** com Transfer Learning em duas fases sobre o dataset público **PlantDoc**.

- **Dataset:** PlantDoc (Kaggle, formato YOLO) — 2.569 imagens, 30 classes, reorganizadas em 1.843 treino / 460 validação / 235 teste.
- **Modelo:** EfficientNetB0 (ImageNet) → GAP → BN → Dense(256) → Dropout → Dense(30, softmax). Loss `categorical_crossentropy`, Adam, batch 32, resolução 224×224, EarlyStopping/ReduceLROnPlateau.
- **Fase 1 (cabeça congelada):** LR=1e-3, 40 épocas → val_acc 76,96%.
- **Fase 2 (fine-tuning, últimas 30 camadas):** LR=1e-4, 33 épocas → val_acc 83,26%.

### Resultados

| Fase | Melhor val_acc | Acc. Teste | Loss Teste |
|---|---|---|---|
| Base congelada | 76,96% | — | — |
| Fine-Tuning | **83,26%** | — | — |
| Modelo final | 83,26% | **56,17%** | 1,7461 |

> Gap validação (83%) vs. teste (56%): as pastas `data/` e `test/` foram capturadas em condições distintas pelo Roboflow, gerando diferença de distribuição — resultado típico reportado para o PlantDoc na literatura.

**Classes com melhor desempenho:** Strawberry leaf (F1 0,94), grape leaf black rot (F1 0,87), Squash Powdery mildew leaf (F1 0,82). **Classes mais difíceis:** Tomato leaf yellow virus (0% de acerto), Bell_pepper leaf spot e Corn Gray leaf spot — a alta similaridade visual entre as 8 classes de doenças de tomate concentra a maior parte dos erros.

### Comparação com a literatura (acurácia no teste do PlantDoc)

| Trabalho | Modelo | Estratégia | Acurácia |
|---|---|---|---|
| Thapa et al. (2020) | VGG16 | Transfer Learning | 54,8% |
| Thapa et al. (2020) | InceptionV3 | Transfer Learning | 50,1% |
| Thapa et al. (2020) | ResNet50 | Transfer Learning | 46,9% |
| **Este trabalho** | **EfficientNetB0** | **TL + Fine-Tuning** | **56,17%** |

O modelo supera os baselines VGG16, InceptionV3 e ResNet50 do artigo original do PlantDoc. Trabalhos futuros: EfficientNetB3/B5 ou ResNet101, balanceamento de classes (class weights/SMOTE), semi-supervised learning e Grad-CAM para interpretabilidade.

---

## 4. Clusterização de Doenças em Plantas (PlantDoc)

Abordagem **não supervisionada** sobre o mesmo dataset PlantDoc: extração de características com **EfficientNetB0** (extrator fixo, sem treino, `include_top=False`, pooling médio) seguida de padronização, **PCA** e **K-Means**, com o número de clusters definido pelo Método do Cotovelo.

- **Total de imagens:** 2.538 · **Dimensão original das features:** 1.280 (EfficientNetB0) · **Componentes PCA (90% variância):** 425.
- **k sugerido pelo cotovelo:** 13 · **k sugerido pela silhueta:** 2 · **k de referência (classes reais):** 30.
- **K-Means final:** k=13, silhueta média −0,0104.

### Comparação clusters × classes reais

| Métrica | Valor |
|---|---|
| Adjusted Rand Index (ARI) | 0,1165 |
| Normalized Mutual Info (NMI) | 0,2947 |
| Homogeneidade | 0,2589 |
| Completude | 0,3420 |
| V-measure | 0,2947 |

Os clusters de maior pureza foram o cluster 3 (62,7% dominado por *Corn leaf blight*) e o cluster 8 (54,3% *Strawberry leaf*). Os valores baixos de ARI/silhueta indicam que as 30 classes reais de doenças não se separam bem apenas com características visuais genéricas da ImageNet — reforçando a necessidade de fine-tuning supervisionado (trabalho 3) para essa tarefa.

Artefatos salvos em [`plantdoc/clusterizacao_resultados/`](plantdoc/clusterizacao_resultados/): `clusters_atribuidos.csv`, `scaler.joblib`, `pca.joblib`, `kmeans_final.joblib`.

---

## 5. LLM — Comparação de Desempenho

Avaliação comparativa de **LLMs locais** (via Ollama) em uma tarefa de **RAG** (Retrieval-Augmented Generation) sobre o *Regulamento de Estágio da UFRB*, medindo qualidade da resposta (ROUGE-1/2/L) e custo/latência de inferência.

- **Modelos avaliados:** `gemma3:4b`, `gemma:2b`, `deepseek-r1:1.5b`.
- **Métricas:** tokens totais, latência (s), tokens/segundo, custo estimado (USD), ROUGE-1/2/L F1.
- **Máquina de teste:** Intel Core i5-1135G7 @ 2.40GHz, 8GB RAM DDR4.

### Pergunta 1 — "Pode ser solicitado dispensa do estágio supervisionado obrigatório?"

| Modelo | Tokens | Latência (s) | Tokens/s | Custo (USD) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| gemma3:4b | 178 | 47,77 | 3,43 | 0,000267 | 0,9005 | 0,8995 | 0,9005 |
| gemma:2b | 46 | 3,54 | 9,04 | 0,000069 | 0,2342 | 0,0550 | 0,1622 |
| deepseek-r1:1.5b | 214 | 56,22 | 3,56 | 0,000321 | 0,4953 | 0,3208 | 0,4486 |

### Pergunta 2 — "Quais documentos são necessários para redução de carga horária?"

| Modelo | Tokens | Latência (s) | Tokens/s | Custo (USD) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| gemma3:4b | 49 | 63,89 | 0,53 | 0,000073 | 0,4222 | 0,3636 | 0,4222 |
| gemma:2b | 50 | 39,97 | 0,88 | 0,000075 | 0,3505 | 0,1263 | 0,2268 |
| deepseek-r1:1.5b | 68 | 135,09 | 0,39 | 0,000102 | 0,0174 | 0,0000 | 0,0174 |

**Conclusão:** o `gemma3:4b` obteve a melhor qualidade de resposta (ROUGE mais alto) na pergunta 1, mas nenhum modelo respondeu corretamente a pergunta 2 (o desempenho é fortemente afetado pelo hardware disponível para o teste, como registrado no próprio experimento).

---

## Estrutura do repositório

```
aprendizado de maquina/
├── README.md                                  (este arquivo)
├── CNN_Deforestation_v4.ipynb                 → Trabalho 1
├── Detecção de Desmatamento.pdf               → Slides do Trabalho 1
├── regressao_gasolina.ipynb                   → Trabalho 2
├── Previsão de Preços de gasolina.pdf         → Slides do Trabalho 2
├── Classificação de Doenças em plantas.pdf    → Slides do Trabalho 3
├── plantdoc/
│   ├── plantdoc_classificacaoVersion2.ipynb   → Trabalho 3
│   ├── plantdoc_clusterizacao_correto.ipynb   → Trabalho 4
│   ├── clusterizacao_resultados/              → Modelos/CSV do Trabalho 4
│   └── (imagens de resultados, dataset PlantDoc em formato YOLO)
└── LLM/
    ├── README.md                              → Detalhes do Trabalho 5
    ├── llm_aula1_RAG.ipynb, llm_metrics.ipynb, atividade.py, metrics.py
    └── Regulamento_Estagio_UFRB.pdf           → Base usada no RAG
```
