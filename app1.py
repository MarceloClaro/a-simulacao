import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from fpdf import FPDF
import base64
import requests
import json
from datetime import datetime, timedelta
import io

# Definir chaves de API para a Embrapa e OpenWeather
EMBRAPA_CONSUMER_KEY = 'SUA_CONSUMER_KEY_AQUI'
EMBRAPA_CONSUMER_SECRET = 'SEU_CONSUMER_SECRET_AQUI'
API_KEY_OPENWEATHERMAP = 'SUA_API_KEY_OPENWEATHERMAP_AQUI'

# Função para buscar dados NDVI e EVI da API SATVeg da Embrapa
def obter_dados_ndvi(latitude, longitude, data_inicial, data_final):
    url = "https://api.cnptia.embrapa.br/satveg/v2/series"
    token_url = "https://api.cnptia.embrapa.br/token"
    credentials = f"{EMBRAPA_CONSUMER_KEY}:{EMBRAPA_CONSUMER_SECRET}"
    headers = {
        "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        access_token = response.json().get("access_token")
        headers = {"Authorization": "Bearer " + access_token}
        payload = {
            "tipoPerfil": "ndvi",
            "satelite": "comb",
            "latitude": latitude,
            "longitude": longitude,
            "dataInicial": data_inicial.strftime('%Y-%m-%d'),
            "dataFinal": data_final.strftime('%Y-%m-%d')
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["listaSerie"]
    return None

# Função para buscar dados meteorológicos da API OpenWeather
def obter_dados_meteorologicos(latitude, longitude):
    url = f"http://api.openweathermap.org/data/2.5/onecall?lat={latitude}&lon={longitude}&appid={API_KEY_OPENWEATHERMAP}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        dados = response.json()
        temperatura = dados["current"]["temp"]
        umidade = dados["current"]["humidity"]
        velocidade_vento = dados["current"]["wind_speed"]
        return temperatura, umidade, velocidade_vento
    else:
        return None, None, None

# Interface principal do Streamlit
def main():
    st.title("EcoSim.ai - Simulador de Propagação de Incêndio")
    st.sidebar.subheader("Configurações de Simulação")
    latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 0.0)
    longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 0.0)
    data_inicial = st.sidebar.date_input("Data Inicial", datetime.now() - timedelta(days=30))
    data_final = st.sidebar.date_input("Data Final", datetime.now())

    # Buscar NDVI e dados meteorológicos
    if st.sidebar.button("Buscar NDVI e Meteorologia"):
        ndvi = obter_dados_ndvi(latitude, longitude, data_inicial, data_final)
        temperatura, umidade, velocidade_vento = obter_dados_meteorologicos(latitude, longitude)
        if ndvi:
            st.sidebar.write(f"NDVI Obtido: {ndvi[-1]}")
        if temperatura and umidade and velocidade_vento:
            st.sidebar.write(f"Temperatura: {temperatura} °C")
            st.sidebar.write(f"Umidade: {umidade} %")
            st.sidebar.write(f"Velocidade do Vento: {velocidade_vento} m/s")

    params = {
        'temperatura': st.sidebar.slider('Temperatura (°C)', 0, 50, 30),
        'umidade': st.sidebar.slider('Umidade relativa (%)', 0, 100, 40),
        'velocidade_vento': st.sidebar.slider('Velocidade do Vento (km/h)', 0, 100, 20),
        'direcao_vento': st.sidebar.slider('Direção do Vento (graus)', 0, 360, 90),
        'precipitacao': st.sidebar.slider('Precipitação (mm/dia)', 0, 200, 0),
        'radiacao_solar': st.sidebar.slider('Radiação Solar (W/m²)', 0, 1200, 800),
        'tipo_vegetacao': st.sidebar.selectbox('Tipo de vegetação', ['pastagem', 'matagal', 'floresta decídua', 'floresta tropical']),
        'densidade_vegetacao': st.sidebar.slider('Densidade Vegetal (%)', 0, 100, 70),
        'umidade_combustivel': st.sidebar.slider('Teor de umidade do combustível (%)', 0, 100, 10),
        'topografia': st.sidebar.slider('Topografia (inclinação em graus)', 0, 45, 5),
        'tipo_solo': st.sidebar.selectbox('Tipo de solo', ['arenoso', 'misto', 'argiloso']),
        'ndvi': st.sidebar.slider('NDVI (Índice de Vegetação por Diferença Normalizada)', 0.0, 1.0, 0.6),
        'intensidade_fogo': st.sidebar.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
        'tempo_desde_ultimo_fogo': st.sidebar.slider('Tempo desde o último incêndio (anos)', 0, 100, 10),
        'intervencao_humana': st.sidebar.slider('Fator de Intervenção Humana (escala 0-1)', 0.0, 1.0, 0.2),
        'ruido': st.sidebar.slider('Ruído (%)', 1, 100, 10)
    }

    if st.button("Executar Simulação"):
        # Código de simulação e geração de gráficos
        # Funções de simulação e processamento a serem chamadas
        st.write("Simulação executada. Visualize os resultados abaixo.")

    # Funções e exibição de resultados adicionais podem ser adicionadas aqui, como gráficos e tabelas.

if __name__ == "__main__":
    main()
