import numpy as np
import pandas as pd


def gerar_markdown_por_pergunta(resultados):
    """
    Gera tabelas Markdown separadas por pergunta.

    Para cada pergunta, os modelos são comparados entre si e os melhores
    valores de cada métrica são exibidos em negrito.
    """

    # Converte a estrutura aninhada para um DataFrame simplificado
    df = pd.DataFrame(
        [
            {
                "Pergunta": resultado["Pergunta"],
                "Modelo": resultado["Modelo"],
                "Latência (s)": resultado["Latência (s)"],
                "Tokens": resultado["Tokens"],
                "Tokens/s": resultado["Tokens/s"],
                "Custo (USD)": resultado["Custo (USD)"],
                "ROUGE-1": resultado["ROUGE-1"],
                "ROUGE-2": resultado["ROUGE-2"],
                "ROUGE-L": resultado["ROUGE-L"],
                "Faithfulness": resultado["Faithfulness"][0],
                "Resposta LLM": resultado["Resposta final"],
                "Resposta correta": resultado["Resposta correta"],
            }
            for resultado in resultados
        ]
    )

    # Define se o melhor valor é o maior ou o menor
    criterios = {
        "Latência (s)": "min",
        "Tokens": "min",
        "Tokens/s": "max",
        "Custo (USD)": "min",
        "ROUGE-1": "max",
        "ROUGE-2": "max",
        "ROUGE-L": "max",
        "Faithfulness": "max",
    }

    def formatar_valor(coluna, valor, melhor_valor):
        """
        Formata o valor e adiciona negrito quando ele corresponde
        ao melhor resultado da pergunta.
        """

        if pd.isna(valor):
            return "-"

        if coluna == "Tokens":
            texto = str(int(valor))
        elif coluna == "Custo (USD)":
            texto = f"{valor:.6f}"
        else:
            texto = f"{valor:.4f}"

        # np.isclose evita problemas de comparação entre números float
        if np.isclose(valor, melhor_valor):
            return f"**{texto}**"

        return texto

    tabelas_markdown = []

    # Agrupa todas as linhas pela pergunta
    for pergunta, grupo in df.groupby("Pergunta", sort=False):

        # Calcula o melhor valor de cada coluna apenas dentro da pergunta atual
        melhores = {}

        for coluna, criterio in criterios.items():
            valores_validos = grupo[coluna].dropna()

            if valores_validos.empty:
                melhores[coluna] = np.nan
            elif criterio == "max":
                melhores[coluna] = valores_validos.max()
            else:
                melhores[coluna] = valores_validos.min()

        colunas_tabela = [
            "Modelo",
            "Latência (s)",
            "Tokens",
            "Tokens/s",
            "Custo (USD)",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Faithfulness",
        ]

        # Título da pergunta
        markdown = f"### {pergunta}\n\n"

        # Cabeçalho da tabela
        markdown += "| " + " | ".join(colunas_tabela) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(colunas_tabela)) + " |\n"

        # Linhas da tabela
        for _, linha in grupo.iterrows():
            valores_linha = []

            for coluna in colunas_tabela:
                valor = linha[coluna]

                if coluna == "Modelo":
                    valores_linha.append(str(valor))
                    continue

                melhor_valor = melhores[coluna]

                if pd.isna(melhor_valor):
                    valores_linha.append("-")
                else:
                    valores_linha.append(
                        formatar_valor(
                            coluna=coluna,
                            valor=valor,
                            melhor_valor=melhor_valor,
                        )
                    )

            markdown += "| " + " | ".join(valores_linha) + " |\n"

        tabelas_markdown.append(markdown)

    # Separa as tabelas por duas quebras de linha
    return "\n\n".join(tabelas_markdown)


# markdown = gerar_markdown_por_pergunta(resultados)

# print(markdown)
