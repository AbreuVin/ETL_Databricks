# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %pip install sentence-transformers
# MAGIC import requests
# MAGIC import google.generativeai as genai
# MAGIC import pandas as pd
# MAGIC import time
# MAGIC import re
# MAGIC from transformers import pipeline
# MAGIC from sentence_transformers import util
# MAGIC

# COMMAND ----------

# DBTITLE 1,Untitled
# MAGIC %pip install -q google-generativeai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

df = spark.sql("""
    SELECT *
    FROM workspace.default.items_licitacoes_2019_2024
""")


# COMMAND ----------

# DBTITLE 1,Cell 3
display(df)

# COMMAND ----------

# DBTITLE 1,Count unique values in descricao
chave_gemini = dbutils.secrets.get(scope="escopo_gemini_", key="chave_agrupamento")
import google.generativeai as genai 
genai.configure(api_key=chave_gemini)

# COMMAND ----------

# DBTITLE 1,Export unique descriptions to CSV
unique_descriptions = df.select('descricao').distinct().toPandas()
unique_descriptions.to_csv('/Workspace/Users/viniciusabreu115@gmail.com/descriptions_unique.csv', index=False)
print('CSV com descrições únicas salvo em /Workspace/Users/viniciusabreu115@gmail.com/descriptions_unique.csv')

# COMMAND ----------

# DBTITLE 1,Cell 8
import pandas as pd
caminho_csv = "/Workspace/Users/viniciusabreu115@gmail.com/descriptions_unique.csv" 
df = pd.read_csv(caminho_csv)

# COMMAND ----------

dados_amostra = df.to_csv(index=False)

# COMMAND ----------

# DBTITLE 1,Cell 11
modelo = genai.GenerativeModel('gemini-2.5-flash')
prompt_fase1 = f"""
Você é um algoritmo de clusterização semântica.
Leia a seguinte lista completa de itens únicos presentes no meu banco de dados:
{dados_amostra}

Tarefa:
- Agrupe TODOS os itens logicamente.
- Crie NO MÁXIMO 25 categorias.
- Cada categoria deve conter uma pares no formato nome_da_categoria: descrição_da_categoria.
- CATEGORIAS DEVEM ser específicas, distintas e sem sobreposição. NÃO use agrupamentos genéricos como "Outros" ou "Diversos".
- FORMATO OBRIGATÓRIO: Retorne APENAS um dicionário Python válido e puro, SEM explicações, comentários, markdown, cabeçalho, ou texto extra antes/depois. EXATAMENTE assim:
{{"Alimentos": "Arroz, feijão, ingredientes de cozinha", "Saúde": "Medicamentos, Seringas", "Materiais de Construção": "Tijolos, Argamassa, Madeira", "Tecnologia": "Celulares, Computadores, Impressoras"}}
- Use apenas aspas duplas, vírgulas para separar pares, e nada antes nem depois.
- NÃO inclua código markdown (nada de ```), cabeçalhos ou qualquer texto além do dicionário.
"""

resposta = modelo.generate_content(prompt_fase1)
categorias_com_contexto = resposta.text


if categorias_com_contexto.startswith('```python'):
    categorias_com_contexto = categorias_com_contexto[len('```python'):]
categorias_com_contexto = categorias_com_contexto.strip()
if categorias_com_contexto.endswith('```'):
    categorias_com_contexto = categorias_com_contexto[:-3]


idx_dict = categorias_com_contexto.find('{')
if idx_dict != -1:
    categorias_com_contexto = categorias_com_contexto[idx_dict:]

print(categorias_com_contexto)

# COMMAND ----------

# DBTITLE 1,Cell 10
import ast
if categorias_com_contexto is not None:
    try:
        parsed_categorias = ast.literal_eval(categorias_com_contexto)
    except Exception as e:
        print("Erro ao interpretar categorias:\n", categorias_com_contexto)
        raise e
    nomes_categorias = list(parsed_categorias.keys())
    descricao = list(parsed_categorias.values())
else:
    print("categorias_com_contexto não está definido ou é None. Execute a célula anterior primeiro.")

# COMMAND ----------

print(nomes_categorias)

# COMMAND ----------

def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# COMMAND ----------

# DBTITLE 1,Cell 14
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("intfloat/multilingual-e5-large")

print("Gerando embeddings das categorias...")
embeddings_categorias = modelo.encode(nomes_categorias, convert_to_tensor=True)
print("Pronto!\n")

# COMMAND ----------

# DBTITLE 1,Cell 13
import ast
parsed_categorias = ast.literal_eval(categorias_com_contexto)
nomes_categorias = list(parsed_categorias.keys())
descricao = list(parsed_categorias.values())

# COMMAND ----------

LIMIAR_CONFIANCA = 0.25  # Ajuste conforme necessário

def classificar(descricao):
    texto_limpo = limpar_texto(descricao)
    if texto_limpo == "":
        return "Sem Descrição", 0.0

    embedding_descricao = modelo.encode(texto_limpo, convert_to_tensor=True)
    scores = util.cos_sim(embedding_descricao, embeddings_categorias)[0]

    melhor_idx = scores.argmax().item()
    melhor_score = scores[melhor_idx].item()
    melhor_categoria = nomes_categorias[melhor_idx]

    if melhor_score >= LIMIAR_CONFIANCA:
        return melhor_categoria, round(melhor_score, 2)
    else:
        return "Outros / Necessita Revisão", round(melhor_score, 2)

# COMMAND ----------

# DBTITLE 1,Cell 16
import re
from sentence_transformers import util

caminho_arquivo = caminho_csv
df = pd.read_csv(caminho_arquivo)

# Teste com 20 primeiros
df_amostra = df.copy()

print(f"Classificando {len(df_amostra)} itens...\n")

resultados = []
for _, linha in df_amostra.iterrows():
    descricao = linha['descricao']
    categoria, confianca = classificar(descricao)

    resultados.append({
        "descricao_original": descricao,
        "categoria_ia": categoria,
        "confianca": confianca,
    })

    print(f"{descricao[:35]:<35} -> {categoria:<35} (Confiança: {confianca*100:.1f}%)")

# COMMAND ----------

df_resultado = pd.DataFrame(resultados)
df_resultado = spark.createDataFrame(df_resultado)
df_resultado.write.mode("overwrite").saveAsTable("mapa_descricao")
print("\n--- Tabela Final ---")
display(df_resultado)