import streamlit as st
import numpy as np
import pandas as pd
import base64
import requests
from datetime import datetime, timedelta

# Definindo as chaves de API (substitua pelas suas próprias)
API_KEY_OPENWEATHERMAP = '5af75d4ed5ae582e673f8d0ba4728936'
EMBRAPA_CONSUMER_KEY = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
EMBRAPA_CONSUMER_SECRET = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Função para obter coordenadas (latitude e longitude) a partir de um endereço ou nome de cidade
def obter_coordenadas_endereco(endereco):
    url = f"https://nominatim.openstreetmap.org/search?q={endereco}&format=json&limit=1"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        resultado = response.json()[0]
        return float(resultado['lat']), float(resultado['lon'])
    else:
        st.error("Endereço não encontrado.")
        return None, None

# Função para obter o token de acesso da Embrapa
def obter_token_acesso_embrapa(consumer_key, consumer_secret):
    token_url = 'https://api.cnptia.embrapa.br/token'
    credentials = f"{consumer_key}:{consumer_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {'grant_type': 'client_credentials'}
    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        token_info = response.json()
        return token_info['access_token']
    else:
        st.error(f"Erro ao obter token da Embrapa: {response.status_code}")
        return None

# Função para obter dados NDVI e EVI da Embrapa usando latitude e longitude
def obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi', satelite='comb'):
    access_token = obter_token_acesso_embrapa(EMBRAPA_CONSUMER_KEY, EMBRAPA_CONSUMER_SECRET)
    if not access_token:
        return None

    url = 'https://api.cnptia.embrapa.br/satveg/v2/series'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    payload = {
        "tipoPerfil": tipo_indice,
        "satelite": satelite,
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

# Interface principal do Streamlit
def main():
    st.title("Simulador de Propagação de Incêndio com Dados NDVI e Meteorológicos")
    
    # Inicializar latitude e longitude
    latitude, longitude = None, None
    
    endereco = st.text_input("Digite a localização (cidade, endereço ou coordenadas):")

    if st.button("Obter Coordenadas"):
        if endereco:
            latitude, longitude = obter_coordenadas_endereco(endereco)
            if latitude and longitude:
                st.write(f"Coordenadas: Latitude {latitude}, Longitude {longitude}")

    # Configuração das datas para a coleta de dados NDVI/EVI
    data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=30))
    data_final = st.date_input("Data Final", datetime.now())

    # Buscar NDVI e EVI
    if latitude is not None and longitude is not None:
        tipo_indice = st.selectbox("Selecione o Índice Vegetativo", ["ndvi", "evi"])
        if st.button("Obter NDVI/EVI"):
            ndvi_evi_series = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice=tipo_indice)
            if ndvi_evi_series is not None:
                st.line_chart(ndvi_evi_series.set_index('Data'))

if __name__ == "__main__":
    main()
