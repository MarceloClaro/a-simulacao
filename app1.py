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

# Funções de obtenção de dados e API
def obter_token_acesso_embrapa(consumer_key, consumer_secret):
    token_url = 'https://api.cnptia.embrapa.br/token'
    credentials = f"{consumer_key}:{consumer_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    headers = {'Authorization': f'Basic {encoded_credentials}', 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(token_url, headers=headers, data={'grant_type': 'client_credentials'})
    return response.json().get('access_token') if response.status_code == 200 else None

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
    return pd.DataFrame({'Data': pd.to_datetime(response.json()['listaDatas']), tipo_indice.upper(): response.json()['listaSerie']})

def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime('%Y-%m-%d'),
        "end_date": data_final.strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
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
        "Velocidade_Vento_10m": hourly.Variables(2).ValuesAsNumpy(),
        "Direcao_Vento_10m": hourly.Variables(3).ValuesAsNumpy(),
    }
    return pd.DataFrame(hourly_data)

def obter_coordenadas_endereco(endereco):
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(endereco)}&format=json&limit=1"
    response = requests.get(url, headers={'User-Agent': 'SimuladorIncendio/1.0'})
    if response.status_code == 200 and response.json():
        resultado = response.json()[0]
        return float(resultado['lat']), float(resultado['lon'])
    st.error("Endereço não encontrado.")
    return None, None

# Funções de simulação de incêndio
VIVO, QUEIMANDO, QUEIMADO = 0, 1, 2

def inicializar_grade(tamanho, inicio_fogo):
    grade = np.full((tamanho, tamanho), VIVO)
    grade[inicio_fogo] = QUEIMANDO
    return grade

def aplicar_regras_fogo(grade, prob_propagacao):
    nova_grade = grade.copy()
    tamanho = grade.shape[0]
    for i in range(1, tamanho - 1):
        for j in range(1, tamanho - 1):
            if grade[i, j] == QUEIMANDO:
                nova_grade[i, j] = QUEIMADO
                vizinhos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in vizinhos:
                    if grade[ni, nj] == VIVO and np.random.rand() < prob_propagacao:
                        nova_grade[ni, nj] = QUEIMANDO
    return nova_grade

def executar_simulacao(tamanho, passos, inicio_fogo, prob_propagacao):
    grade = inicializar_grade(tamanho, inicio_fogo)
    grades = [grade.copy()]
    for _ in range(passos):
        grade = aplicar_regras_fogo(grade, prob_propagacao)
        grades.append(grade.copy())
    return grades

def plotar_simulacao(grades):
    num_graficos = min(10, len(grades))  # Exibir no máximo 10 gráficos
    indices = np.linspace(0, len(grades) - 1, num_graficos, dtype=int)
    
    fig, axes = plt.subplots(1, num_graficos, figsize=(15, 5))
    cmap = ListedColormap(['green', 'red', 'black'])
    
    for ax, idx in zip(axes, indices):
        ax.imshow(grades[idx], cmap=cmap, interpolation='nearest')
        ax.set_title(f'Passo {idx}')
        ax.axis('off')
    
    st.pyplot(fig)

# Interface do usuário
def main():
    st.title("Simulador de Propagação de Incêndio")
    st.subheader("Automação de Parâmetros Usando APIs")

    st.header("Seleção de Localização e Período de Análise")
    endereco = st.text_input("Digite a localização (ex.: cidade, endereço):")
    
    if st.button("Buscar Coordenadas"):
        latitude, longitude = obter_coordenadas_endereco(endereco) if endereco else (None, None)
        if latitude and longitude:
            st.success(f"Coordenadas encontradas: Latitude {latitude}, Longitude {longitude}")
            st.session_state['latitude'] = latitude
            st.session_state['longitude'] = longitude
    
    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        latitude = st.session_state['latitude']
        longitude = st.session_state['longitude']

        data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=7))
        data_final = st.date_input("Data Final", datetime.now())

        if st.button("Obter Dados Meteorológicos e Índices de Vegetação"):
            hourly_df = obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final)
            ndvi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final, tipo_indice='ndvi')

            st.write("### Dados Meteorológicos")
            st.write(hourly_df)
            st.write("### Índice NDVI")
            st.write(ndvi_df)

            if not hourly_df.empty and not ndvi_df.empty:
                prob_propagacao = hourly_df['Temperatura_2m'].mean() * 0.01
                st.subheader("Configurações da Simulação")
                tamanho_grade = st.slider("Tamanho da Grade", 10, 100, 50)
                passos = st.slider("Número de Passos da Simulação", 10, 200, 100)
                inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)

                if st.button("Executar Simulação de Incêndio"):
                    simulacao = executar_simulacao(tamanho_grade, passos, inicio_fogo, prob_propagacao)
                    plotar_simulacao(simulacao)

if __name__ == '__main__':
    main()
