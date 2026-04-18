# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

df = spark.table("workspace.default.licitacoes_2019_2024")

# COMMAND ----------

# DBTITLE 1,Estatísticas descritivas (describe)
display(df.describe())

# COMMAND ----------

# DBTITLE 1,Filtrar apenas colunas relevantes para série temporal
# Lista de colunas desejadas
colunas_relevantes = [
    'valor', 'descricao', 'descricao_original', 'categoria_ia', 
    'mes/ano', 'quantidade', 'nome', 
    'descUnidadeFornecimento', 'descComplementarItemCompra'
]

# Verifica colunas existentes no DataFrame
colunas_presentes = [col for col in colunas_relevantes if col in df.columns]
colunas_faltantes = [col for col in colunas_relevantes if col not in df.columns]

# Cria novo DataFrame apenas com colunas disponíveis
if colunas_presentes:
    df_selected = df.select(*colunas_presentes)
    display(df_selected.limit(20))
    if colunas_faltantes:
        print(f'Atenção: as seguintes colunas não estão presentes: {colunas_faltantes}')
else:
    print('Nenhuma das colunas relevantes está presente no DataFrame.')

# COMMAND ----------

# DBTITLE 1,Comportamento das 25 categorias mais frequentes ao longo do tempo
# Converter a base selecionada para pandas (limitada a 50000 ou menos linhas para evitar crash)
pdf = df_selected.limit(50000).toPandas()

# Identificar as 25 categorias mais frequentes
categorias = pdf['categoria_ia'].value_counts().head(25).index.tolist()

# Filtrar apenas esses top 25
pdf_top = pdf[pdf['categoria_ia'].isin(categorias)].copy()

# Converter mes/ano para datetime
pdf_top['mes_ano_dt'] = pd.to_datetime(pdf_top['mes/ano'], format='%m/%Y', errors='coerce')

# Agrupar e calcular soma dos valores por categoria e por mês
grupo = pdf_top.groupby(['categoria_ia', 'mes_ano_dt'])['valor'].sum().reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18,10))
for categoria in categorias:
    serie = grupo[grupo['categoria_ia'] == categoria].sort_values('mes_ano_dt')
    plt.plot(serie['mes_ano_dt'], serie['valor'], label=categoria)
plt.title('Evolução dos valores nas 25 categorias mais frequentes')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Evolução mensal do valor total (todas categorias)
pdf = df_selected.toPandas()
pdf['valor'] = pdf['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
pdf['valor'] = pd.to_numeric(pdf['valor'], errors='coerce')
pdf['mes_ano_dt'] = pd.to_datetime(pdf['mes/ano'], format='%m/%Y', errors='coerce')

# Selecionar só as linhas válidas
geral = pdf.dropna(subset=['valor', 'mes_ano_dt'])

# Agregar valor total por mês (todas categorias)
valor_mensal_total = geral.groupby('mes_ano_dt')['valor'].sum().reset_index()

plt.figure(figsize=(16,9))
plt.plot(valor_mensal_total['mes_ano_dt'], valor_mensal_total['valor'], marker='o', color='blue')
plt.title('Evolução mensal do valor total (todas categorias)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

print('\nEstatísticas descritivas do valor total mensal:')
print(valor_mensal_total['valor'].describe())

# COMMAND ----------

# DBTITLE 1,Fluxo de análise temporal para uma categoria específica
# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[0]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando regex
import re
# Remove tudo menos digitos e vírgulas/pontos
df_categoria['valor'] = df_categoria['valor'].astype(str).apply(lambda x: re.sub(r'[^\d,\.]', '', x))
# Troca vírgula decimal por ponto
df_categoria['valor'] = df_categoria['valor'].str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# DBTITLE 1,Evolução mensal dos valores - Top 10 categorias
# Analise mensal de todas as categorias (top 10)
pdf = df_selected.toPandas()
pdf['valor'] = pdf['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
pdf['valor'] = pd.to_numeric(pdf['valor'], errors='coerce')
pdf['mes_ano_dt'] = pd.to_datetime(pdf['mes/ano'], format='%m/%Y', errors='coerce')

# Selecionar só as linhas válidas
pdf = pdf.dropna(subset=['valor', 'categoria_ia', 'mes_ano_dt'])

# Agregar valor mensal por categoria
grupo = pdf.groupby(['categoria_ia','mes_ano_dt'])['valor'].sum().reset_index()

# Selecionar top 10 categorias por volume total
top_categorias = grupo.groupby('categoria_ia')['valor'].sum().nlargest(10).index.tolist()
grupo_filtrado = grupo[grupo['categoria_ia'].isin(top_categorias)]

plt.figure(figsize=(16,9))
for categoria in top_categorias:
    serie = grupo_filtrado[grupo_filtrado['categoria_ia'] == categoria]
    plt.plot(serie['mes_ano_dt'], serie['valor'], marker='o', label=categoria)
plt.title('Evolução mensal dos valores - Top 10 categorias')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Fluxo de análise temporal para uma categoria específica
# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[0]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando replace, lidando com ponto e vírgula
# Remove pontos (milhar), troca vírgula decimal por ponto, então converte para float
df_categoria['valor'] = df_categoria['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# DBTITLE 1,Decomposição da série: tendência, sazonalidade e ruído
# Decomposição da série temporal: tendência, sazonalidade e ruído
from statsmodels.tsa.seasonal import seasonal_decompose

# Usar a série temporal agregada mensal (grupo_categoria_mensal)
serie_temporal = grupo_categoria_mensal.set_index('mes_ano_dt')['valor']

# Realizar decomposição (frequência mensal)
decomp = seasonal_decompose(serie_temporal, model='additive', period=12)

plt.figure(figsize=(14,6))
decomp.trend.plot(title='Tendência da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.seasonal.plot(title='Sazonalidade da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.resid.plot(title='Ruído/Resíduos da Série', legend=True)
plt.show()

# COMMAND ----------

# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[1]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando replace, lidando com ponto e vírgula
# Remove pontos (milhar), troca vírgula decimal por ponto, então converte para float
df_categoria['valor'] = df_categoria['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# Decomposição da série temporal: tendência, sazonalidade e ruído
from statsmodels.tsa.seasonal import seasonal_decompose

# Usar a série temporal agregada mensal (grupo_categoria_mensal)
serie_temporal = grupo_categoria_mensal.set_index('mes_ano_dt')['valor']

# Realizar decomposição (frequência mensal)
decomp = seasonal_decompose(serie_temporal, model='additive', period=12)

plt.figure(figsize=(14,6))
decomp.trend.plot(title='Tendência da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.seasonal.plot(title='Sazonalidade da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.resid.plot(title='Ruído/Resíduos da Série', legend=True)
plt.show()

# COMMAND ----------

# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[2]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando replace, lidando com ponto e vírgula
# Remove pontos (milhar), troca vírgula decimal por ponto, então converte para float
df_categoria['valor'] = df_categoria['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# Decomposição da série temporal: tendência, sazonalidade e ruído
from statsmodels.tsa.seasonal import seasonal_decompose

# Usar a série temporal agregada mensal (grupo_categoria_mensal)
serie_temporal = grupo_categoria_mensal.set_index('mes_ano_dt')['valor']

# Realizar decomposição (frequência mensal)
decomp = seasonal_decompose(serie_temporal, model='additive', period=12)

plt.figure(figsize=(14,6))
decomp.trend.plot(title='Tendência da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.seasonal.plot(title='Sazonalidade da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.resid.plot(title='Ruído/Resíduos da Série', legend=True)
plt.show()

# COMMAND ----------

# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[3]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando replace, lidando com ponto e vírgula
# Remove pontos (milhar), troca vírgula decimal por ponto, então converte para float
df_categoria['valor'] = df_categoria['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# Decomposição da série temporal: tendência, sazonalidade e ruído
from statsmodels.tsa.seasonal import seasonal_decompose

# Usar a série temporal agregada mensal (grupo_categoria_mensal)
serie_temporal = grupo_categoria_mensal.set_index('mes_ano_dt')['valor']

# Realizar decomposição (frequência mensal)
decomp = seasonal_decompose(serie_temporal, model='additive', period=12)

plt.figure(figsize=(14,6))
decomp.trend.plot(title='Tendência da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.seasonal.plot(title='Sazonalidade da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.resid.plot(title='Ruído/Resíduos da Série', legend=True)
plt.show()

# COMMAND ----------

# Garantir que "pdf" está definido
pdf = df_selected.toPandas()

# ETAPA 1: Visualizar categorias disponíveis
categorias_disponiveis = pdf['categoria_ia'].unique()
print('Categorias disponíveis em categoria_ia:')
for cat in categorias_disponiveis:
    print(cat)

# ETAPA 2: Escolher uma categoria para análise
categoria_escolhida = categorias_disponiveis[4]  # Troque para a que desejar
print(f'Categoria escolhida: {categoria_escolhida}')

# ETAPA 3: Filtrar o DataFrame para a categoria selecionada
df_categoria = pdf[pdf['categoria_ia'] == categoria_escolhida].copy()

# ETAPA 3.1: Converter coluna 'valor' para float usando replace, lidando com ponto e vírgula
# Remove pontos (milhar), troca vírgula decimal por ponto, então converte para float
df_categoria['valor'] = df_categoria['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df_categoria['valor'] = pd.to_numeric(df_categoria['valor'], errors='coerce')

# ETAPA 4: Converter mes/ano para datetime
df_categoria['mes_ano_dt'] = pd.to_datetime(df_categoria['mes/ano'], format='%m/%Y', errors='coerce')

# ETAPA 5: Agrupar e calcular soma dos valores por mês
# Remove valores ausentes e garante apenas meses válidos
grupo_categoria_mensal = df_categoria.groupby('mes_ano_dt')['valor'].sum().reset_index()

# ETAPA 6: Plotar evolução temporal mensal
plt.figure(figsize=(14, 6))
plt.plot(grupo_categoria_mensal['mes_ano_dt'], grupo_categoria_mensal['valor'], marker='o')
plt.title(f'Evolução mensal dos valores para categoria: {categoria_escolhida} (6 anos)')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor total mensal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extra: Estatísticas básicas da categoria
print('\nEstatísticas descritivas da categoria:')
print(df_categoria['valor'].describe())

# COMMAND ----------

# Decomposição da série temporal: tendência, sazonalidade e ruído
from statsmodels.tsa.seasonal import seasonal_decompose

# Usar a série temporal agregada mensal (grupo_categoria_mensal)
serie_temporal = grupo_categoria_mensal.set_index('mes_ano_dt')['valor']

# Realizar decomposição (frequência mensal)
decomp = seasonal_decompose(serie_temporal, model='additive', period=12)

plt.figure(figsize=(14,6))
decomp.trend.plot(title='Tendência da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.seasonal.plot(title='Sazonalidade da Série', legend=True)
plt.show()
plt.figure(figsize=(14,6))
decomp.resid.plot(title='Ruído/Resíduos da Série', legend=True)
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

