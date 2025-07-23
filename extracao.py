import xarray as xr
import pandas as pd
import sys
import os
import zipfile

# O arquivo que você já tem, que na verdade é um ZIP
output_filename_zip = "dados_junho_rs.nc"
output_filename_csv = "dados_rs18.csv"

# Etapa 1: Descompactar o arquivo (seu código já faz isso bem)
try:
    with open(output_filename_zip, 'rb') as f:
        signature = f.read(2)
    
    if signature == b'PK':
        print(f"Arquivo '{output_filename_zip}' é um arquivo ZIP. Descompactando...")
        with zipfile.ZipFile(output_filename_zip, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            print(f"Arquivos encontrados no ZIP: {extracted_files}")
            zip_ref.extractall('.')
            print("Arquivos extraídos com sucesso.")
    else:
        # Se não for ZIP, a lista de arquivos contém apenas o nome original
        extracted_files = [output_filename_zip]

except Exception as e:
    print(f"ERRO AO PROCESSAR O ARQUIVO: {e}", file=sys.stderr)
    sys.exit(1)

# --- Etapa 2: Abrir e JUNTAR os múltiplos arquivos NetCDF (CORRIGIDO) ---
try:
    # Identifica os dois arquivos extraídos
    instant_file = next((f for f in extracted_files if 'instant' in f), None)
    accum_file = next((f for f in extracted_files if 'accum' in f), None)

    if not instant_file or not accum_file:
        raise FileNotFoundError("Não foi possível encontrar os arquivos 'instant' e 'accum' no ZIP.")

    print(f"Abrindo arquivo instantâneo: '{instant_file}'")
    ds_instant = xr.open_dataset(instant_file)

    print(f"Abrindo arquivo acumulado: '{accum_file}'")
    ds_accum = xr.open_dataset(accum_file)

    # Junta (merge) os dois datasets em um só
    print("Juntando os datasets de dados instantâneos e acumulados...")
    ds = xr.merge([ds_instant, ds_accum])
    print("Datasets juntados com sucesso!")

    print("\n--- Conteúdo do Dataset Carregado ---")
    print(ds)
    print("-------------------------------------\n")
    # -------------------------------------------

except Exception as e:
    print(f"ERRO CRÍTICO: Falha ao abrir e juntar os arquivos NetCDF.", file=sys.stderr)
    print(f"Erro específico: {e}", file=sys.stderr)
    sys.exit(1)


# --- Etapa 3: Processamento dos dados (agora com o dataset completo) ---
print("Convertendo os dados para um formato de tabela (DataFrame)...")
df = ds.to_dataframe().reset_index()

print("Criando colunas de data, precipitação e temperatura...")
# A coluna de tempo do ERA5 geralmente se chama 'time'
df['data'] = df['valid_time'].dt.date
df['precipitacao_mm'] = df['tp'] * 1000
df['temperatura'] = df['t2m'] - 273.15

# O resto do seu código de agregação...
aggregations = {
    'precipitacao_total_mm':  ('precipitacao_mm', 'sum'),
    'temperatura_media_C':    ('temperatura', 'mean'),
    'temperatura_max_C':      ('temperatura', 'max'),
    'temperatura_min_C':      ('temperatura', 'min'),
    'pressao_Pa' :            ('sp', 'mean'),
    'vento_u_10m' :           ('u10','mean'),
    'vento_v_10m' :           ('v10', 'mean')
}

agregacoes_finais = {}
for nome_nova_coluna, (nome_coluna_original, operacao) in aggregations.items():
    if nome_coluna_original in df.columns:
        agregacoes_finais[nome_nova_coluna] = (nome_coluna_original, operacao)

if agregacoes_finais:
    df_diario = df.groupby(['latitude', 'longitude', 'data']).agg(**agregacoes_finais).reset_index()
    print("Resultado do Agrupamento (primeiras 5 linhas):")
    print(df_diario.head())
else:
    print("Nenhuma coluna para agregar foi encontrada.")

print(f"Salvando como arquivo CSV: {output_filename_csv}")
df_diario.to_csv(output_filename_csv, index=False, sep=',', decimal='.')

print("\nSucesso!")