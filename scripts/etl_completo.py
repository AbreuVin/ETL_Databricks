import os
import requests
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, sum, desc, regexp_replace
from delta import *

# 1. Configuração do Spark com Delta Lake (Simulando Databricks)
builder = SparkSession.builder \
    .appName("ETL_Gov_Local") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
    .master("local[*]")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR") # Menos poluição no terminal

# --- CAMADA BRONZE (Ingestão da API) ---
print("\n>>> INICIANDO CAMADA BRONZE...")

api_key = os.getenv("API_TOKEN")
if not api_key:
    raise ValueError("API_TOKEN não encontrado no .env")

# Vamos pegar dados de despesas por órgão (apenas página 1 para teste rápido)
url = "https://api.portaldatransparencia.gov.br/api-v1/despesas/por-orgao"
headers = {"chave-api-dados": api_key}
params = {"ano": 2024, "pagina": 1}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    dados = response.json()
    df_raw = spark.createDataFrame(dados)
    
    # Adiciona metadados
    df_bronze = df_raw.withColumn("data_ingestao", current_timestamp())
    
    # Salva em formato DELTA (como no Databricks)
    path_bronze = "/app/data/bronze/despesas"
    df_bronze.write.format("delta").mode("overwrite").save(path_bronze)
    print(f"✅ Dados salvos na Bronze: {path_bronze}")
else:
    print(f"❌ Erro na API: {response.status_code}")
    exit()

# --- CAMADA SILVER (Limpeza) ---
print("\n>>> INICIANDO CAMADA SILVER...")

# Lê da Bronze
df_b = spark.read.format("delta").load(path_bronze)

# Transforma
df_silver = (df_b
    .withColumnRenamed("codigoOrgao", "cod_orgao")
    .withColumnRenamed("nomeOrgao", "nome_orgao")
    .withColumnRenamed("ano", "ano_exercicio")
    # Limpa valor monetário e converte para double
    .withColumn("valor_limpo", regexp_replace(col("valor"), ",", ".").cast("double"))
    .select("ano_exercicio", "cod_orgao", "nome_orgao", "valor_limpo")
)

path_silver = "/app/data/silver/despesas_limpas"
df_silver.write.format("delta").mode("overwrite").save(path_silver)
print(f"✅ Dados limpos salvos na Silver: {path_silver}")

# --- CAMADA GOLD (Agregação) ---
print("\n>>> INICIANDO CAMADA GOLD...")

# Lê da Silver
df_s = spark.read.format("delta").load(path_silver)

# Agrega: Total gasto por Órgão
df_gold = (df_s
    .groupBy("nome_orgao")
    .agg(sum("valor_limpo").alias("total_gasto"))
    .orderBy(desc("total_gasto"))
)

path_gold = "/app/data/gold/resumo_gastos"
df_gold.write.format("delta").mode("overwrite").save(path_gold)

print("✅ Tabela Gold gerada com sucesso!")
print("\n--- TOP 5 ÓRGÃOS QUE MAIS GASTARAM ---")
df_gold.show(5, truncate=False)

spark.stop()