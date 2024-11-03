import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import base64
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon
from fpdf import FPDF
import urllib.parse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io

# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="Simulador de Propagação de Incêndio",
    page_icon="🔥",
    layout="wide"
)

# Definição dos estados das células na simulação
VIVO = 0        # Vegetação não queimada
QUEIMANDO = 1   # Vegetação em chamas
QUEIMADO = 2    # Vegetação já queimada
RECUPERADO = 3  # Vegetação recuperada após o incêndio

# Cores associadas a cada estado para visualização
colors = {
    VIVO: 'green',
    QUEIMANDO: 'red',
    QUEIMADO: 'black',
    RECUPERADO: 'blue'
}

# Função para obter dados meteorológicos usando Open-Meteo API
def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    # Configurar URL e parâmetros da API
    data_inicial_str = data_inicial.strftime('%Y-%m-%d')
    data_final_str = data_final.strftime('%Y-%m-%d')
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&"
        f"start_date={data_inicial_str}&end_date={data_final_str}&"
        f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    
    # Requisição GET
    response = requests.get(url)
    if response.status_code == 200:
        dados = response.json()
        
        # Extrair dados horários
        horas = dados['hourly']['time']
        temperaturas = dados['hourly']['temperature_2m']
        umidades = dados['hourly']['relative_humidity_2m']
        velocidades_vento = dados['hourly']['wind_speed_10m']

        # Criar DataFrame com os dados extraídos
        df = pd.DataFrame({
            'Data': pd.to_datetime(horas),
            'Temperatura': temperaturas,
            'Umidade': umidades,
            'Vento': velocidades_vento
        })
        return df
    else:
        st.warning(f"Erro ao obter dados meteorológicos: {response.status_code}")
        return None

# Função para obter coordenadas de um endereço
def obter_coordenadas_endereco(endereco):
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(endereco)}&format=json&limit=1"
    headers = {'User-Agent': 'SimuladorIncendio/1.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        resultado = response.json()[0]
        return float(resultado['lat']), float(resultado['lon'])
    else:
        st.error("Endereço não encontrado.")
        return None, None

# Interface do usuário
def main():
    st.title("Simulador de Propagação de Incêndio")
    st.subheader("Automação de Parâmetros Usando APIs")

    # Variáveis iniciais
    meteo_series = None

    # Seleção de localização
    st.header("Seleção de Localização")
    endereco = st.text_input("Digite a localização (cidade, endereço ou coordenadas):")

    if st.button("Obter Coordenadas"):
        if endereco:
            lat, lon = obter_coordenadas_endereco(endereco)
            if lat and lon:
                st.success(f"Coordenadas: Latitude {lat}, Longitude {lon}")
                st.session_state['latitude'] = lat
                st.session_state['longitude'] = lon
        else:
            st.error("Por favor, insira um endereço válido.")

    # Se coordenadas foram obtidas
    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        lat = st.session_state['latitude']
        lon = st.session_state['longitude']

        # Configuração do período de análise
        st.header("Período de Análise")
        data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=7))
        data_final = st.date_input("Data Final", datetime.now())

        # Dados Meteorológicos
        st.header("Dados Meteorológicos")
        if st.button("Obter Dados Meteorológicos"):
            meteo_series = obter_dados_meteorologicos(lat, lon, data_inicial, data_final)
            if meteo_series is not None:
                st.line_chart(meteo_series.set_index('Data'))

if __name__ == '__main__':
    main()
