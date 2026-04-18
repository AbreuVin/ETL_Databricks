# Databricks notebook source
# DBTITLE 1,Cell 1
import pandas as pd
import requests
import re
import json
import base64
import calendar
import time
from pyspark.sql.functions import current_timestamp, lit, col, regexp_replace, sum, desc
from databricks.sdk.runtime import dbutils

# COMMAND ----------

chave_api = dbutils.secrets.get(scope="transparencia-scope", key="api-token")
chave_decodificada = base64.b64decode(chave_api)

# COMMAND ----------

# DBTITLE 1,Cell 3
def obter_dados_api(codigo_orgao, data_inicial, data_final):
    url = "https://api.portaldatransparencia.gov.br/api-de-dados/licitacoes"
    params = {"codigoOrgao": codigo_orgao, "dataInicial": data_inicial, "dataFinal": data_final, "pagina": 1}
    headers = {"accept": "application/json", "chave-api-dados": chave_api}
    dados_paginas = []
    while True:
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            dados_json = response.json()
            if not dados_json:
                break
            dados_paginas.extend(dados_json)
            params["pagina"] += 1
        except requests.exceptions.RequestException as e:
            print("Erro ao fazer a requisição:", e)
            break
    return dados_paginas

df = pd.DataFrame()
def criar_dataframe(codigo_orgao, data_inicial, data_final):
    global df
    dados_contratos = obter_dados_api(codigo_orgao, data_inicial, data_final)
    if dados_contratos:
        df = pd.DataFrame(dados_contratos)
        return df
    else:
        return None

# COMMAND ----------

list_df_mes = []
for mes in range(1, 13):
    codigo_orgao = "52111"
    ano = 2021
    ultimo_dia = calendar.monthrange(ano, mes)[1]
    mes_str = str(mes).zfill(2)
    data_inicial = f"01/{mes_str}/{ano}"          
    data_final = f"{ultimo_dia}/{mes_str}/{ano}"
    df_temporario = criar_dataframe(codigo_orgao, data_inicial, data_final)
    if df_temporario is not None:
        df_temporario.insert(0,"mes/ano",f"{mes_str}/{ano}")
        list_df_mes.append(df_temporario)
        print(list_df_mes)

# COMMAND ----------

# DBTITLE 1,Cell 5
df = pd.concat(list_df_mes, ignore_index=True)
hashable_types = (str, int, float, bool, type(None))
cols_hashable = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, hashable_types)).all()]
df_hashable = df[cols_hashable]
num_df_duplicadas = df_hashable.duplicated().sum()
print(f"Total de linhas duplicadas: {num_df_duplicadas}")

# COMMAND ----------

len(df)

# COMMAND ----------

# DBTITLE 1,Cell 7
def obter_dados_api2(id):
    url = "https://api.portaldatransparencia.gov.br/api-de-dados/licitacoes/itens-licitados"

    params = {"id": id, "pagina": 1}
    headers = {"accept": "application/json", "chave-api-dados": chave_api}

    dados_paginas = []

    while True:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            dados_json = response.json()

            if not dados_json:
                break

            dados_paginas.extend(dados_json)
            print(f"Página: {params['pagina']}, Itens retornados: {len(dados_json)}")
            params["pagina"] += 1

    return dados_paginas


def criar_dataframe2(id):      
    dados_contratos = obter_dados_api2(id)

    if dados_contratos:
        df2 = pd.DataFrame(dados_contratos)
        return df2
    else:
        return None

# COMMAND ----------

# DBTITLE 1,Add id_licitacao to df2 safely
id_list = []
for id_licitacao in df["id"]:
    try:
        id_licitacao = str(id_licitacao)
        df2 = criar_dataframe2(id_licitacao)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            df2["id"] = id_licitacao
            id_list.append(df2)
    except: 
        print(f"Erro ao processar o ID: {id_licitacao}")

# COMMAND ----------

df2_consolidado = pd.concat(id_list, ignore_index=True)

# COMMAND ----------

df2_consolidado = df2_consolidado.merge(df[['id','mes/ano']].assign(id=df['id'].astype(str)), how='left', on='id')

# COMMAND ----------

df2_consolidado

# COMMAND ----------

df2_consolidado["id"]

# COMMAND ----------

num_duplicadas = df2_consolidado.duplicated().sum()
print(f"Total de linhas duplicadas: {num_duplicadas}")

# COMMAND ----------

df2_consolidado_sem_duplicatas = df2_consolidado.drop_duplicates()

# COMMAND ----------

df2_consolidado_sem_duplicatas

# COMMAND ----------

# DBTITLE 1,Write items_licitacoes table with matching schema
spark_df_licitacoes = spark.createDataFrame(df2_consolidado_sem_duplicatas)
spark_df_licitacoes.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("items_licitacoes_2021")

# COMMAND ----------

spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").saveAsTable("licitacoes_2021")