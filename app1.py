import streamlit as st
import numpy as np
import pandas as pd
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

# Chaves de API (substitua pelas suas próprias)
EMBRAPA_CONSUMER_KEY = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
EMBRAPA_CONSUMER_SECRET = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Configuração de cache e sessões de requisições com retry
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Função para obter token de acesso da Embrapa
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

# Função para obter NDVI e EVI da Embrapa
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

# Função para obter dados meteorológicos usando Open-Meteo API com cache
def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime('%Y-%m-%d'),
        "end_date": data_final.strftime('%Y-%m-%d'),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", 
            "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit", 
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", 
            "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", 
            "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", 
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", 
            "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
            "sunrise", "sunset", "daylight_duration", "sunshine_duration", 
            "precipitation_sum", "rain_sum", "precipitation_hours", 
            "wind_speed_10m_max", "wind_gusts_10m_max", 
            "wind_direction_10m_dominant", "shortwave_radiation_sum", 
            "et0_fao_evapotranspiration"
        ]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Processamento de dados horários
    hourly = response.Hourly()
    hourly_data = {
        "Data": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "Temperatura_2m": hourly.Variables(0).ValuesAsNumpy(),
        "Umidade_Relativa_2m": hourly.Variables(1).ValuesAsNumpy(),
        "Temperatura_Aparente": hourly.Variables(2).ValuesAsNumpy(),
        "Pressao_Superficie": hourly.Variables(3).ValuesAsNumpy(),
        "Cobertura_Nuvens": hourly.Variables(4).ValuesAsNumpy(),
        "Cobertura_Nuvens_Baixa": hourly.Variables(5).ValuesAsNumpy(),
        "Cobertura_Nuvens_Media": hourly.Variables(6).ValuesAsNumpy(),
        "Cobertura_Nuvens_Alta": hourly.Variables(7).ValuesAsNumpy(),
        "Evapotranspiracao_ET0_FAO": hourly.Variables(8).ValuesAsNumpy(),
        "Deficit_Pressao_Vapor": hourly.Variables(9).ValuesAsNumpy(),
        "Velocidade_Vento_10m": hourly.Variables(10).ValuesAsNumpy(),
        "Velocidade_Vento_100m": hourly.Variables(11).ValuesAsNumpy(),
        "Direcao_Vento_10m": hourly.Variables(12).ValuesAsNumpy(),
        "Direcao_Vento_100m": hourly.Variables(13).ValuesAsNumpy(),
        "Rajadas_Vento_10m": hourly.Variables(14).ValuesAsNumpy(),
        "Temperatura_Solo_0_7cm": hourly.Variables(15).ValuesAsNumpy(),
        "Temperatura_Solo_7_28cm": hourly.Variables(16).ValuesAsNumpy(),
        "Temperatura_Solo_28_100cm": hourly.Variables(17).ValuesAsNumpy(),
        "Temperatura_Solo_100_255cm": hourly.Variables(18).ValuesAsNumpy(),
        "Umidade_Solo_0_7cm": hourly.Variables(19).ValuesAsNumpy(),
        "Umidade_Solo_7_28cm": hourly.Variables(20).ValuesAsNumpy(),
        "Umidade_Solo_28_100cm": hourly.Variables(21).ValuesAsNumpy(),
        "Umidade_Solo_100_255cm": hourly.Variables(22).ValuesAsNumpy()
    }
    hourly_df = pd.DataFrame(hourly_data)

    # Processamento de dados diários
    daily = response.Daily()
    daily_data = {
        "Data": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "Temperatura_Maxima_2m": daily.Variables(0).ValuesAsNumpy(),
        "Temperatura_Minima_2m": daily.Variables(1).ValuesAsNumpy(),
        "Temperatura_Media_2m": daily.Variables(2).ValuesAsNumpy(),
        "Temperatura_Aparente_Maxima": daily.Variables(3).ValuesAsNumpy(),
        "Temperatura_Aparente_Minima": daily.Variables(4).ValuesAsNumpy(),
        "Temperatura_Aparente_Media": daily.Variables(5).ValuesAsNumpy(),
        "Nascer_do_Sol": daily.Variables(6).ValuesAsNumpy(),
        "Por_do_Sol": daily.Variables(7).ValuesAsNumpy(),
        "Duracao_do_Dia": daily.Variables(8).ValuesAsNumpy(),
        "Duracao_do_Sol": daily.Variables(9).ValuesAsNumpy(),
        "Precipitacao_Total": daily.Variables(10).ValuesAsNumpy(),
        "Chuva_Total": daily.Variables(11).ValuesAsNumpy(),
        "Horas_de_Precipitacao": daily.Variables(12).ValuesAsNumpy(),
        "Velocidade_Maxima_Vento_10m": daily.Variables(13).ValuesAsNumpy(),
        "Rajadas_Maximas_Vento_10m": daily.Variables(14).ValuesAsNumpy(),
        "Direcao_Dominante_Vento_10m": daily.Variables(15).ValuesAsNumpy(),
        "Radiacao_Curta_Total": daily.Variables(16).ValuesAsNumpy(),
        "Evapotranspiracao_Total": daily.Variables(17).ValuesAsNumpy()
    }
    daily_df = pd.DataFrame(daily_data)
    
    return hourly_df, daily_df

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

# Interface do usuário
def main():
    st.title("Simulador de Propagação de Incêndio")
    st.subheader("Automação de Parâmetros Usando APIs")

    # Seleção de localização e período
    st.header("Seleção de Localização e Período de Análise")
    endereco = st.text_input("Digite a localização (ex.: cidade, endereço):")
    
    if st.button("Buscar Coordenadas"):
        if endereco:
            latitude, longitude = obter_coordenadas_endereco(endereco)
            if latitude and longitude:
                st.success(f"Coordenadas encontradas: Latitude {latitude}, Longitude {longitude}")
                st.session_state['latitude'] = latitude
                st.session_state['longitude'] = longitude
        else:
            st.error("Por favor, insira uma localização válida.")
    
    # Utilização de coordenadas armazenadas
    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        latitude = st.session_state['latitude']
        longitude = st.session_state['longitude']

        data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=14))
        data_final = st.date_input("Data Final", datetime.now())

        if st.button("Obter Dados Meteorológicos e Índices de Vegetação"):
            hourly_df, daily_df = obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final)
            ndvi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi')
            evi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='evi')
            
            st.write("### Dados Meteorológicos Horários")
            st.write(hourly_df)
            
            st.write("### Dados Meteorológicos Diários")
            st.write(daily_df)
            
            st.write("### Índice NDVI")
            st.write(ndvi_df)
            
            st.write("### Índice EVI")
            st.write(evi_df)

if __name__ == '__main__':
    main()
