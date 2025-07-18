import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geobr

def spectral_clustering(data: np.ndarray, k: int, sigma: float):
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

# --- 1. Carregar os Dados ---
try:
    ds = xr.open_dataset("precip_rs_jun172025.grib", engine="cfgrib",
                         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# --- 1.5. CALCULAR E PLOTAR A SOMA TOTAL DA PREVISÃO ---
print("Calculando a precipitação total da primeira rodada de previsão...")

# <-- CORREÇÃO DEFINITIVA AQUI
# 1. Seleciona a primeira rodada da previsão (time=0)
forecast_run = ds['tp'].isel(time=0)
# 2. Agora, soma a chuva de todos os 'steps' (12h) DENTRO dessa rodada
total_precip_periodo = forecast_run.sum(dim='step')
#----------------------------------

total_precip_periodo_mm = total_precip_periodo * 1000 # Converte para mm

# Cria o primeiro gráfico
plt.figure(figsize=(10, 8.5))
ax1 = plt.axes()
total_precip_periodo_mm.plot.imshow(ax=ax1, cmap='inferno', cbar_kwargs={'label': 'Precipitação (mm)'})
ax1.set_title("Precipitação Total Acumulada (12h da 1ª Previsão)")
print("Mostrando o mapa de chuva total. Feche a janela para continuar para o clustering.")
plt.show()

# --- 2. Preparar os Dados para a Clusterização (para UM horário) ---
print("\nPreparando dados para o clustering do meio-dia...")
# Para o clustering, selecionamos um único tempo E passo (ex: previsão de 6h da primeira rodada)
precip_data_2d = ds['tp'].isel(time=0, step=11) * 1000

lats = precip_data_2d.latitude.values
lons = precip_data_2d.longitude.values
precips = precip_data_2d.values

lon_grid, lat_grid = np.meshgrid(lons, lats)
feature_data = np.stack([lat_grid.flatten(), lon_grid.flatten(), precips.flatten()], axis=1)
valid_indices = ~np.isnan(feature_data).any(axis=1)
feature_data_valid = feature_data[valid_indices, :]

# --- 3. Normalização e Ponderação ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data_valid)
precipitation_weight = 5.0
scaled_features[:, 2] *= precipitation_weight

# --- 4. Executar o Algoritmo de Cluster ---
k = 4
sigma = 1.0
print(f"Executando Spectral Clustering para k={k}...")
labels = spectral_clustering(scaled_features, k=k, sigma=sigma)
print("Clusterização concluída.")

# --- 5. Plotar o Mapa de Clusters ---
final_labels = np.full(precips.shape, np.nan)
valid_indices_2d = np.unravel_index(np.where(valid_indices)[0], precips.shape)
final_labels[valid_indices_2d] = labels

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.set_title(f'Clusters de Precipitação (Previsão de 6h)', fontsize=16)
im = ax2.pcolormesh(lons, lats, final_labels, cmap='viridis', shading='auto')

try:
    rs_shape = geobr.read_state(code_state='RS', year=2020)
    rs_shape.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1.5, zorder=10)
except Exception as e:
    print(f"Não foi possível baixar o mapa do RS: {e}")

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cbar = fig.colorbar(im, ax=ax2, ticks=range(k))
cbar.set_label('Cluster ID')
ax2.grid(True, linestyle='--', alpha=0.5)

print("Mostrando o mapa de clusters.")
plt.show()