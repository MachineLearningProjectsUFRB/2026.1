import time
import json
import tiktoken

from rouge_score import rouge_scorer
from langchain_ollama import ChatOllama

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
from rouge_score import rouge_scorer # type: ignore

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
    
def llmAsJudge(pergunta, resposta, gabarito, judge):

    llm = ChatOllama(
        model=judge,
        temperature=0
    )

    prompt = f"""
Você é um avaliador imparcial.

Compare a resposta da IA com o gabarito.

PERGUNTA:
{pergunta}

GABARITO:
{gabarito}

RESPOSTA DA IA:
{resposta}

Avalie:

1. Correção (0 a 10)

2. Completude (0 a 10)

3. Clareza (0 a 10)

Depois calcule uma nota final.

Responda APENAS em JSON.

Formato:

{{
"correcao":0,
"completude":0,
"clareza":0,
"nota_final":0,
"comentario":""
}}

"""

    resposta_llm = llm.invoke(prompt)

    texto = resposta_llm.content.strip()

    print("========== RESPOSTA DO JUIZ ==========")
    print(texto)
    print("======================================")
    texto = texto.replace("```json", "")
    texto = texto.replace("```", "")
    texto = texto.strip()

    resultado = json.loads(texto)
    
    resultado.setdefault("correcao", 0)
    resultado.setdefault("completude", 0)
    resultado.setdefault("clareza", 0)
    resultado.setdefault("nota_final", 0)
    resultado.setdefault("comentario", "")

    return resultado