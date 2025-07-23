import cdsapi
import calendar
import os
import sys

# ==============================================================================
# --- CONFIGURAÇÃO ---
# Altere apenas os valores nesta seção

# 1. Ano que você deseja baixar
ANO_ALVO = 2022

# 2. Pasta onde os 12 arquivos mensais serão salvos
PASTA_SAIDA = "dados_era5_2022"

# 3. Lista de variáveis que você quer baixar
VARIAVEIS = [
    '2m_temperature',
    'total_precipitation',
    'surface_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]

# 4. Área geográfica de interesse [Norte, Oeste, Sul, Leste]
AREA = [-27, -58, -34, -49]

# ==============================================================================


def baixar_dados_mensais(cliente_cds, ano, mes):
    """
    Função para baixar os dados de um mês específico do ERA5.
    """
    nome_arquivo = f"dados_ERA5_{ano}_{mes:02d}.nc"
    caminho_completo = os.path.join(PASTA_SAIDA, nome_arquivo)

    if os.path.exists(caminho_completo):
        # Versão sem emoji
        print(f"[OK] Arquivo '{nome_arquivo}' ja existe. Pulando...")
        return True

    _, num_dias = calendar.monthrange(ano, mes)
    lista_dias = [f'{dia:02d}' for dia in range(1, num_dias + 1)]

    # Versão sem emoji
    print(f"--> Iniciando download para {mes:02d}/{ano}...")

    try:
        cliente_cds.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': VARIAVEIS,
                'year': str(ano),
                'month': f'{mes:02d}',
                'day': lista_dias,
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                         '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                         '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                         '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'area': AREA,
                'format': 'netcdf',
            },
            caminho_completo
        )
        # Versão sem emoji
        print(f"[SUCESSO] Download para {mes:02d}/{ano} concluido. Salvo como '{nome_arquivo}'.")
        return True
    except Exception as e:
        # Versão sem emoji
        print(f"[ERRO] Falha ao baixar dados para {mes:02d}/{ano}.", file=sys.stderr)
        print(f"   Motivo: {e}", file=sys.stderr)
        print(f"   O script continuara para o proximo mes.", file=sys.stderr)
        return False


# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(PASTA_SAIDA):
        print(f"Criando pasta de saida: '{PASTA_SAIDA}'")
        os.makedirs(PASTA_SAIDA)

    c = cdsapi.Client()

    print(f"\n--- INICIANDO DOWNLOAD ANUAL PARA {ANO_ALVO} ---")

    for mes_alvo in range(1, 13):
        print("-" * 50)
        baixar_dados_mensais(c, ANO_ALVO, mes_alvo)

    print("\n--- PROCESSO DE DOWNLOAD ANUAL FINALIZADO ---")
    print(f"Todos os arquivos foram salvos na pasta: '{PASTA_SAIDA}'")