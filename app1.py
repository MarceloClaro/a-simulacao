import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from fpdf import FPDF
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime, timedelta
import base64

# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="Simulador de Propagação de Incêndio",
    page_icon="🔥",
    layout="wide"
)

# Funções para obter dados meteorológicos e índices de vegetação
def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime('%Y-%m-%d'),
        "end_date": data_final.strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json()['hourly'])
    else:
        st.error(f"Erro ao obter dados meteorológicos: {response.status_code}")
        return None

def obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi'):
    url = 'https://api.cnptia.embrapa.br/satveg/v2/series'
    headers = {'Authorization': f'Bearer {token}'}
    payload = {
        "tipoPerfil": tipo_indice,
        "latitude": latitude,
        "longitude": longitude,
        "dataInicial": data_inicial.strftime('%Y-%m-%d'),
        "dataFinal": data_final.strftime('%Y-%m-%d')
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        series = pd.DataFrame({
            'Data': pd.to_datetime(data['listaDatas']),
            tipo_indice.upper(): data['listaSerie']
        })
        return series
    else:
        st.error(f"Erro ao obter NDVI/EVI: {response.status_code}")
        return None

# Função para calcular probabilidade de propagação com base nos dados
def calcular_probabilidade_propagacao(params):
    probabilidade_base = 0.3
    
    # Ajustes baseados nos parâmetros
    fator_ndvi = params['ndvi'] * 0.3  # Quanto maior o NDVI, mais combustível disponível
    fator_evi = (1 - params['evi']) * 0.2  # Valores altos de EVI indicam vegetação úmida, dificultando propagação
    fator_umidade = (100 - params['umidade']) / 100 * 0.2  # Menor umidade, maior chance de propagação
    fator_temperatura = (params['temperatura'] - 20) / 30 * 0.2  # Temperaturas mais altas aumentam a propagação
    fator_vento = params['velocidade_vento'] / 50 * 0.3  # Ventos fortes favorecem a propagação

    probabilidade = probabilidade_base + fator_ndvi + fator_evi + fator_umidade + fator_temperatura + fator_vento
    return min(max(probabilidade, 0), 1)  # Limitar entre 0 e 1

# Função para executar simulação usando autômatos celulares
def aplicar_regras_fogo(grade, params):
    nova_grade = grade.copy()
    tamanho = grade.shape[0]
    prob_propagacao = calcular_probabilidade_propagacao(params)

    for i in range(1, tamanho - 1):
        for j in range(1, tamanho - 1):
            if grade[i, j] == 1:  # Células em combustão
                nova_grade[i, j] = 2  # Marca como queimado
                vizinhos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in vizinhos:
                    if grade[ni, nj] == 0 and np.random.rand() < prob_propagacao:
                        nova_grade[ni, nj] = 1  # Propaga o fogo
    return nova_grade

# Função principal para configurar parâmetros e executar simulação
def main():
    st.title("Simulador de Propagação de Incêndio")
    
    # Entrada para localização
    endereco = st.text_input("Digite a localização:")
    latitude, longitude = 0, 0  # Substituir com chamada a uma função que obtenha coordenadas
    
    # Período da simulação
    data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=7))
    data_final = st.date_input("Data Final", datetime.now())
    
    # Carregar dados meteorológicos e NDVI/EVI
    if st.button("Obter Dados"):
        dados_meteo = obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final)
        dados_ndvi = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi')
        dados_evi = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='evi')
        
        if dados_meteo is not None and dados_ndvi is not None and dados_evi is not None:
            # Definindo parâmetros com base nos dados obtidos
            params = {
                'temperatura': dados_meteo['temperature_2m'].mean(),
                'umidade': dados_meteo['relative_humidity_2m'].mean(),
                'velocidade_vento': dados_meteo['wind_speed_10m'].mean(),
                'direcao_vento': dados_meteo['wind_direction_10m'].mean(),
                'ndvi': dados_ndvi['NDVI'].mean(),
                'evi': dados_evi['EVI'].mean(),
                'intensidade_fogo': st.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
                'intervencao_humana': st.slider('Intervenção Humana (0-1)', 0.0, 1.0, 0.2),
                'ruido': st.slider('Ruído (%)', 1, 100, 10)
            }

            # Executa a simulação com os parâmetros ajustados
            tamanho_grade = st.slider("Tamanho da grade", 10, 100, 50)
            passos = st.slider("Número de passos", 10, 200, 100)
            inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
            
            # Inicializa a grade e executa simulação passo a passo
            grade = np.zeros((tamanho_grade, tamanho_grade), dtype=int)
            grade[inicio_fogo] = 1
            simulacao = [grade]
            
            for _ in range(passos):
                grade = aplicar_regras_fogo(grade, params)
                simulacao.append(grade.copy())
            
            # Exibe gráficos da simulação
            cmap = ListedColormap(['green', 'red', 'black'])
            fig, axes = plt.subplots(5, 10, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, grade in enumerate(simulacao[::max(1, len(simulacao)//50)]):
                if i >= len(axes):
                    break
                axes[i].imshow(grade, cmap=cmap, interpolation='nearest')
                axes[i].set_title(f'Passo {i}')
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
