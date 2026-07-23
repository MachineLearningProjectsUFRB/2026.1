# Aprendizado de Máquina — UFRB

Repositório com os trabalhos desenvolvidos pela **Equipe 8** nas disciplinas **Aprendizado de Máquina** e **Tópicos Especiais em Inteligência Artificial**, oferecidas, respectivamente, no Mestrado em Engenharia Elétrica e Computação e no Bacharelado em Engenharia de Computação do CETEC/UFRB.

| Equipe | Docente | Semestre |
| --- | --- | --- |
| Gleice Souza, Lucas Batista e Matheus Mendes | Dr. Tiago Palma Pagano | 2026.1 |

## Trabalhos

| Módulo | Tema | Abordagem | Materiais |
| --- | --- | --- | --- |
| 1 | Classificação de culturas agrícolas | CNNs, aprendizado por transferência e aumento de dados | [Notebook](M%C3%B3dulo%201/Classifica%C3%A7%C3%A3o/Classifica%C3%A7%C3%A3o%20de%20culturas%20agr%C3%ADcolas.ipynb) · [Apresentação](M%C3%B3dulo%201/Classifica%C3%A7%C3%A3o/Classifica%C3%A7%C3%A3o%20de%20culturas%20agr%C3%ADcolas.pdf) |
| 1 | Previsão da velocidade do vento | MLP e diferentes representações de séries temporais | [Notebook](M%C3%B3dulo%201/S%C3%A9ries%20temporais/Projeto%20Mucuri.ipynb) · [Apresentação](M%C3%B3dulo%201/S%C3%A9ries%20temporais/Projeto%20Mucuri.pdf) |
| 2 | Segmentação comportamental de clientes | Engenharia de atributos, PCA e K-Means | [Notebook](M%C3%B3dulo%202/Dunnhumby_The_Complete_Journey.ipynb) · [Apresentação](M%C3%B3dulo%202/Dunnhumby%20-%20The%20Complete%20Journey.pdf) |
| 3 | Avaliação de LLMs locais com RAG | Ollama, Chroma, ROUGE e LLM como juiz | [Notebook](M%C3%B3dulo%203/llm_metrics.ipynb) |

### Módulo 1 — Aprendizado supervisionado

#### Classificação de culturas agrícolas

Compara arquiteturas pré-treinadas para classificar imagens de 30 culturas agrícolas em um cenário com poucos dados. O experimento avalia **EfficientNetB0**, **MobileNetV2**, **ResNet50V2**, **DenseNet121** e **NASNetMobile**, com variações no uso de validação e aumento de dados.

O relatório destaca a EfficientNetB0 como o melhor modelo geral, com **83,62% de acurácia**, **94,35% de acurácia top-3** e **83,18% de F1-score** no conjunto de teste. O notebook contém o comando do Kaggle para baixar uma versão do conjunto de dados previamente dividida em treino e teste.

#### Séries temporais — Projeto Mucuri

Investiga a previsão da velocidade do vento uma hora à frente com uma rede neural MLP. São comparadas três formas de representação dos dados:

- discretização simples;
- codificação trigonométrica de variáveis cíclicas;
- janelamento temporal com diferentes quantidades de atrasos.

O pipeline realiza divisão temporal, normalização, treinamento com parada antecipada e avaliação por MAE, MSE, RMSE, erro máximo, R² e correlação de Pearson. O arquivo CSV com os dados meteorológicos não está incluído no repositório.

### Módulo 2 — Aprendizado não supervisionado

O trabalho utiliza as tabelas de transações e produtos do conjunto de dados **Dunnhumby — The Complete Journey** para segmentar clientes. A análise constrói atributos de recência, frequência, gasto total e percentual de desconto, padroniza os dados e aplica o K-Means. A PCA é utilizada para visualizar os grupos.

Foram identificados três perfis — clientes ocasionais ou em risco, clientes fiéis de alto valor e clientes VIP — com **índice de silhueta (Silhouette Score) de 0,448**.

### Módulo 3 — Avaliação de LLMs

Compara modelos locais em um fluxo RAG baseado em documentos da UFRB. O notebook recupera trechos de uma base vetorial Chroma, gera respostas com os modelos disponíveis no Ollama e mede:

- latência, quantidade de tokens, taxa de processamento e custo estimado;
- ROUGE-1, ROUGE-2 e ROUGE-L;
- fidelidade das respostas, utilizando o modelo `llama3.1:8b` como juiz.

O script `format_output.py` organiza os resultados em tabelas Markdown e destaca o melhor valor de cada métrica por pergunta.

## Estrutura do repositório

```text
.
├── Módulo 1/
│   ├── Classificação/
│   │   ├── Classificação de culturas agrícolas.ipynb
│   │   └── Classificação de culturas agrícolas.pdf
│   └── Séries temporais/
│       ├── Projeto Mucuri.ipynb
│       └── Projeto Mucuri.pdf
├── Módulo 2/
│   ├── Dunnhumby_The_Complete_Journey.ipynb
│   └── Dunnhumby - The Complete Journey.pdf
├── Módulo 3/
│   ├── llm_metrics.ipynb
│   ├── metrics.py
│   └── format_output.py
└── README.md
```

## Como executar

### 1. Prepare o ambiente

Crie um ambiente virtual com Python 3 e instale o Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip jupyterlab
jupyter lab
```

Como alternativa, importe os notebooks no Google Colab.

### 2. Configure os caminhos

Antes da execução, ajuste em cada notebook os diretórios e caminhos utilizados para acessar os dados.
