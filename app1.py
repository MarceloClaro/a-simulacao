import streamlit as st
import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime, timedelta
import base64
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Configuração de cache e sessão de requisições com retry
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Função para obter token de acesso da Embrapa
def obter_token_acesso_embrapa(consumer_key, consumer_secret):
    token_url = 'https://api.cnptia.embrapa.br/token'
    credentials = f"{consumer_key}:{consumer_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {'Authorization': f'Basic {encoded_credentials}', 'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'client_credentials'}
    response = requests.post(token_url, headers=headers, data=data)
    
    if response.status_code == 200:
        token_info = response.json()
        return token_info['access_token']
    else:
        st.error(f"Erro ao obter token da Embrapa: {response.status_code}")
        return None

# Função para obter NDVI e EVI da Embrapa
def obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi', satelite='comb'):
    access_token = obter_token_acesso_embrapa(EMBRAPA_CONSUMER_KEY, EMBRAPA_CONSUMER_SECRET)
    if not access_token:
        return None

    url = 'https://api.cnptia.embrapa.br/satveg/v2/series'
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
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
        series = pd.DataFrame({'Data': pd.to_datetime(data['listaDatas']), tipo_indice.upper(): data['listaSerie']})
        return series
    else:
        st.error(f"Erro ao obter NDVI/EVI: {response.status_code}")
        return None

# Função para obter dados meteorológicos usando Open-Meteo API com cache
def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime('%Y-%m-%d'),
        "end_date": data_final.strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "surface_pressure", "cloud_cover",
                   "wind_speed_10m", "wind_speed_100m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm"]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "Data": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"),
        "Temperatura_2m": hourly.Variables(0).ValuesAsNumpy(),
        "Umidade_Relativa_2m": hourly.Variables(1).ValuesAsNumpy(),
        "Temperatura_Aparente": hourly.Variables(2).ValuesAsNumpy(),
        "Pressao_Superficie": hourly.Variables(3).ValuesAsNumpy(),
        "Cobertura_Nuvens": hourly.Variables(4).ValuesAsNumpy(),
        "Velocidade_Vento_10m": hourly.Variables(5).ValuesAsNumpy(),
        "Velocidade_Vento_100m": hourly.Variables(6).ValuesAsNumpy(),
        "Temperatura_Solo_0_7cm": hourly.Variables(7).ValuesAsNumpy(),
        "Temperatura_Solo_7_28cm": hourly.Variables(8).ValuesAsNumpy()
    }
    hourly_df = pd.DataFrame(hourly_data)
    return hourly_df

# Função para obter coordenadas de uma localidade usando Nominatim
def obter_coordenadas_endereco(endereco):
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(endereco)}&format=json&limit=1"
    headers = {'User-Agent': 'SimuladorIncendio/1.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        resultado = response.json()[0]
        return float(resultado['lat']), float(resultado['lon'])
    else:
        st.error("Endereço não encontrado.")
        return None, None

# Plotando histogramas e gráficos de margem de erro
def plotar_histogramas_e_erros(simulacao):
    contagem_queimando = [np.sum(grade == 1) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(contagem_queimando_df, x="Células Queimando", ax=ax[0], kde=True, bins=20, color='orange')
    ax[0].set_title('Histograma de Células Queimando')
    ax[0].set_xlabel('Número de Células Queimando')
    ax[0].set_ylabel('Frequência')

    media_movel = contagem_queimando_df.rolling(window=10).mean()
    std_movel = contagem_queimando_df.rolling(window=10).std()
    ax[1].plot(media_movel, label='Média', color='blue')
    ax[1].fill_between(std_movel.index, media_movel["Células Queimando"] - std_movel["Células Queimando"], media_movel["Células Queimando"] + std_movel["Células Queimando"], color='blue', alpha=0.2, label='Margem de Erro (1 std)')
    ax[1].set_title('Média e Margem de Erro')
    ax[1].set_xlabel('Passos da Simulação')
    ax[1].set_ylabel('Número de Células Queimando')
    ax[1].legend()

    plt.tight_layout()
    st.sidebar.pyplot(fig)

# Interface do usuário
def main():
    st.title("Simulador de Propagação de Incêndio")
    endereco = st.text_input("Digite a localização (ex.: cidade, endereço):")
    
    if st.button("Buscar Coordenadas"):
        latitude, longitude = obter_coordenadas_endereco(endereco)
        if latitude and longitude:
            st.session_state['latitude'] = latitude
            st.session_state['longitude'] = longitude

    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        latitude, longitude = st.session_state['latitude'], st.session_state['longitude']
        data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=7))
        data_final = st.date_input("Data Final", datetime.now())

        hourly_df = obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final)
        ndvi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi')
        evi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='evi')
        
        if hourly_df is not None and ndvi_df is not None and evi_df is not None:
            st.write("### Dados Meteorológicos (Open-Meteo)")
            st.write(hourly_df)

            st.write("### Índice NDVI (Embrapa)")
            st.write(ndvi_df)

            st.write("### Índice EVI (Embrapa)")
            st.write(evi_df)

            params = {
                'temperatura': hourly_df['Temperatura_2m'].mean(),
                'umidade': hourly_df['Umidade_Relativa_2m'].mean(),
                'vento_10m': hourly_df['Velocidade_Vento_10m'].mean(),
                'ndvi': ndvi_df['NDVI'].mean(),
                'evi': evi_df['EVI'].mean(),
                'chuva': hourly_df['Cobertura_Nuvens'].mean()
            }

            st.write("### Configurações da Simulação")
            tamanho_grade = st.slider("Tamanho da Grade", 10, 100, 50)
            passos = st.slider("Número de Passos da Simulação", 10, 200, 100)
            inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
            ruido = st.slider("Nível de Ruído", 1, 100, 10)

            if st.button("Executar Simulação de Incêndio"):
                simulacao = executar_simulacao(tamanho_grade, passos, inicio_fogo, params, ruido)
                plotar_histogramas_e_erros(simulacao)

if __name__ == '__main__':
    main()
