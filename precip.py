import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geobr

# A sua função de clustering não muda, pois ela opera em um array NumPy
def spectral_clustering(data: np.ndarray, k: int, sigma: float):
    # (O código da sua função de clusterização continua aqui, sem alterações)
    n = data.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist2 = np.dot(data[i] - data[j], data[i] - data[j])
            w = np.exp(-dist2 / (2 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w
    D_inv_sqrt = np.diag(1 / np.sqrt(A.sum(axis=1) + 1e-9))
    L = D_inv_sqrt @ A @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[-k:]
    X = eigvecs[:, idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Y = X / (norms + 1e-9)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(Y)
    return kmeans.labels_

# --- 1. LER DADOS DIÁRIOS DO CSV ---
NOME_ARQUIVO_CSV = "precipitacao_rs.csv" # Coloque aqui o nome do seu arquivo
try:
    print(f"Lendo o arquivo de dados diários: {NOME_ARQUIVO_CSV}")
    # Usamos os separadores corretos para o seu formato
    df = pd.read_csv(NOME_ARQUIVO_CSV, sep=',', decimal='.')
    df['data'] = pd.to_datetime(df['data']).dt.date
except Exception as e:
    print(f"Erro ao carregar o arquivo CSV: {e}")
    exit()

# --- 2. SELECIONAR UM ÚNICO DIA PARA ANÁLISE ---
# Como o arquivo pode ter vários dias, precisamos escolher um para visualizar e clusterizar.
# Vamos pegar o primeiro dia disponível no arquivo como exemplo.
if df['data'].nunique() > 1:
    print(f"Aviso: O arquivo contém {df['data'].nunique()} dias diferentes.")

dia_para_analise = df['data'].unique()[0]
print(f"Analisando dados para o dia: {dia_para_analise}")

# Filtra o DataFrame para conter apenas os dados do dia selecionado.
df_dia_unico = df[df['data'] == dia_para_analise].copy()


# --- 3. PLOTAR O MAPA DE PRECIPITAÇÃO TOTAL DO DIA ---
print(f"Preparando o mapa de precipitação para o dia {dia_para_analise}...")

# MUDANÇA: Não precisamos mais do groupby. Usamos .pivot() para criar a grade.
try:
    precip_grid = df_dia_unico.pivot(index='latitude', columns='longitude', values='precipitacao_total_diaria_mm')
except Exception as e:
    print(f"Erro ao remodelar os dados para o gráfico. Verifique se há dados duplicados para o mesmo dia e local. Erro: {e}")
    exit()

# Garante que as coordenadas são numéricas
lats = precip_grid.index.astype(float)
lons = precip_grid.columns.astype(float)

plt.figure(figsize=(10, 8.5))
ax1 = plt.axes()
im1 = ax1.pcolormesh(lons, lats, precip_grid.values, cmap='inferno', shading='auto')
plt.colorbar(im1, ax=ax1, label='Precipitação Total Diária (mm)')
ax1.set_title(f"Precipitação Total Acumulada em {dia_para_analise}")
print("Mostrando o mapa de chuva. Feche a janela para continuar.")
plt.show()


# --- 4. PREPARAR OS DADOS DO DIA PARA A CLUSTERIZAÇÃO ---
print(f"\nPreparando dados para o clustering do dia {dia_para_analise}...")

# MUDANÇA: A criação do `feature_data` usa o DataFrame do dia único e a coluna correta.
feature_data_valid = df_dia_unico[['latitude', 'longitude', 'precipitacao_total_diaria_mm']].values


# --- 5. NORMALIZAÇÃO E PONDERAÇÃO ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data_valid)
precipitation_weight = 5.0
scaled_features[:, 2] *= precipitation_weight


# --- 6. EXECUTAR O ALGORITMO DE CLUSTER ---
k = 4
sigma = 1.0
print(f"Executando Spectral Clustering para k={k}...")
labels = spectral_clustering(scaled_features, k=k, sigma=sigma)
print("Clusterização concluída.")


# --- 7. PLOTAR O MAPA DE CLUSTERS ---
# Adicionamos os labels ao DataFrame do dia único
df_dia_unico.loc[:, 'cluster'] = labels

# Usamos .pivot() para transformar a tabela de volta em uma grade 2D de clusters.
cluster_grid = df_dia_unico.pivot(index='latitude', columns='longitude', values='cluster')
lats_cluster = cluster_grid.index.astype(float)
lons_cluster = cluster_grid.columns.astype(float)

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.set_title(f'Clusters de Precipitação ({dia_para_analise})', fontsize=16)
im2 = ax2.pcolormesh(lons_cluster, lats_cluster, cluster_grid.values, cmap='viridis', shading='auto')

try:
    rs_shape = geobr.read_state(code_state='RS', year=2020)
    rs_shape.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1.5, zorder=10)
except Exception as e:
    print(f"Não foi possível baixar o mapa do RS: {e}")

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cbar = fig.colorbar(im2, ax=ax2, ticks=range(k))
cbar.set_label('Cluster ID')
ax2.grid(True, linestyle='--', alpha=0.5)

print("Mostrando o mapa de clusters.")
plt.show()