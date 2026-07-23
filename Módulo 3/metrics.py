import time
import tiktoken

def calcular_metricas_operacionais(prompt, resposta, tempo_inicio, tempo_fim, custo_por_1k_tokens=0.0015):
    # Inicializa o contador de tokens (usando a codificação do GPT-4 como exemplo)
    encoder = tiktoken.get_encoding("cl100k_base")
    
    # 1. Contagem de Tokens
    tokens_prompt = len(encoder.encode(prompt))
    tokens_resposta = len(encoder.encode(resposta))
    tokens_totais = tokens_prompt + tokens_resposta
    
    # 2. Latência (Tempo total)
    latencia = tempo_fim - tempo_inicio
    
    # 3. Throughput (Tokens de resposta por segundo)
    tokens_por_segundo = tokens_resposta / latencia if latencia > 0 else 0
    
    # 4. Custo Estimado
    custo = (tokens_totais / 1000) * custo_por_1k_tokens
    
    return {
        "tokens_totais": tokens_totais,
        "latencia_segundos": round(latencia, 2),
        "tokens_por_segundo": round(tokens_por_segundo, 2),
        "custo_estimado_usd": round(custo, 6)
    }


# Necessário instalar: pip install rouge-score
from rouge_score import rouge_scorer

def calcular_rouge(resposta_llm, resposta_referencia):
    # Inicializa o avaliador para ROUGE-1, ROUGE-2 e ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(resposta_referencia, resposta_llm)
    
    # Retorna o F1-Score (balanço entre precisão e recall) de cada um
    return {
        "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
        "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
        "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
    }
    
    
#pip install ragas
#pip install langchain-google-vertexai

# from langchain_ollama import ChatOllama
# from ragas.llms import LangchainLLMWrapper
# from ragas import EvaluationDataset
# from ragas.metrics import Faithfulness
# from ragas import evaluate


import sys
from types import ModuleType

# Cria um módulo falso em memória para simular o caminho que o Ragas procura
mock_vertex = ModuleType("langchain_community.chat_models.vertexai")
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex
mock_vertex.ChatVertexAI = None  # Define o componente interno que o Ragas tenta ler como nulo

# --- Agora seus imports normais vão funcionar sem quebrar ---
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import Faithfulness
from ragas import evaluate

def llmAsJudge(pergunta, resposta, gabarito, judge):
    # 1. Inicializa o ChatOllama apenas com o modelo (sem base_url)
    llm_ollama = ChatOllama(
        model=judge, # "deepseek-r1:8b", # ou llama3.1
        temperature=0
    )

    # 2. Envolve o modelo no padrão exigido pelo Ragas
    juiz_ragas = LangchainLLMWrapper(llm_ollama)

    # 3. Define a métrica usando o seu juiz limpo
    metrica_fidelidade = Faithfulness(llm=juiz_ragas)
    
    # 4. Dados para o teste
    dados = [{
        "user_input": pergunta,
        "response": resposta,
        "retrieved_contexts": [gabarito]
    }]
    dataset = EvaluationDataset.from_list(dados)
    
    # 5. Executa a avaliação
    resultado = evaluate(
        dataset=dataset,
        metrics=[metrica_fidelidade]
    )
    
    return resultado

