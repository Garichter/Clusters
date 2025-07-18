import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import geobr

# A sua função spectral_clustering permanece a mesma, está correta.
def spectral_clustering(data: np.ndarray, k: int, sigma: float):
    n = data.shape[0]

    # Etapa 1: criação de A
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist2 = np.dot(data[i] - data[j], data[i] - data[j])
            w = np.exp(-dist2 / (2 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w

    # Etapa 2: criação da matriz L
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


# --- 1. Carregar os Dados de Precipitação ---
try:
    ds = xr.open_dataset("temp_rs_jun2025.grib", engine="cfgrib")
    # Vamos usar apenas o primeiro horário como exemplo
    # <-- CORREÇÃO: Converte a precipitação de metros para milímetros (mm).
    precip_data = ds['t2m'].isel(time=0) -273
except FileNotFoundError:
    print("Arquivo 'precip_rs_jun2025.grib' não encontrado.")
    print("Certifique-se de que o script de download foi executado com sucesso.")
    exit()
except KeyError:
    print("A variável 'tp' não foi encontrada no arquivo GRIB.")
    print("Verifique as variáveis disponíveis no seu arquivo.")
    exit()


# --- 2. Preparar os Dados para a Clusterização ---

lats = precip_data.latitude.values
lons = precip_data.longitude.values
precips = precip_data.values

# Cria uma grade de coordenadas
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Achata a grade e a precipitação em uma lista de pontos
# Cada ponto será um array [latitude, longitude, precipitação]
n_points = precip_data.size
feature_data = np.zeros((n_points, 3))
feature_data[:, 0] = lat_grid.flatten()
feature_data[:, 1] = lon_grid.flatten()
feature_data[:, 2] = precips.flatten()

# Remove pontos com dados 'nan' (se houver)
valid_indices = ~np.isnan(feature_data).any(axis=1)
feature_data = feature_data[valid_indices, :]


# --- 3. Normalização e Ponderação das Features ---

# Normaliza os dados para que lat, lon e precipitação tenham a mesma escala
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data)

# (Opcional) Aumenta o peso da precipitação.
# <-- CORREÇÃO: Renomeado para 'precipitation_weight' para maior clareza.
precipitation_weight = 2.0
scaled_features[:, 2] *= precipitation_weight


# --- 4. Executar o Algoritmo e Visualizar ---
k = 4
sigma = 1.0

print(f"Executando Spectral Clustering para k={k} e sigma={sigma}...")
labels = spectral_clustering(scaled_features, k=k, sigma=sigma)
print("Clusterização concluída.")

final_labels = np.full(n_points, np.nan)
final_labels[valid_indices] = labels

label_grid = final_labels.reshape(precips.shape)


# --- 5. Plotar o Mapa de Clusters ---

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# <-- CORREÇÃO: Título do gráfico atualizado.
ax.set_title(f'Clusters de Precipitação (k={k})', fontsize=16)

im = ax.pcolormesh(lons, lats, label_grid, cmap='viridis', shading='auto')

try:
    rs_shape = geobr.read_state(code_state='RS', year=2020)
    rs_shape.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=10)
except Exception as e:
    print(f"Não foi possível baixar o mapa do RS. Verifique sua conexão ou a biblioteca geobr. Erro: {e}")

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

cbar = fig.colorbar(im, ax=ax, ticks=range(k))
cbar.set_label('Cluster ID')
ax.grid(True, linestyle='--', alpha=0.5)

plt.show()