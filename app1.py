import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import requests
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Chaves de API (substitua pelas suas próprias)
EMBRAPA_CONSUMER_KEY = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
EMBRAPA_CONSUMER_SECRET = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Configuração do cliente da API Open-Meteo com cache e retry
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Função para obter coordenadas de uma localidade usando Nominatim
def obter_coordenadas_endereco(endereco):
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(endereco)}&format=json&limit=1"
    headers = {'User-Agent': 'SimuladorIncendio/1.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        resultado = response.json()[0]
        return float(resultado['lat']), float(resultado['lon'])
    else:
        st.error("Endereço não encontrado ou fora da América do Sul.")
        return None, None

# Função para obter dados meteorológicos
def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime('%Y-%m-%d'),
        "end_date": data_final.strftime('%Y-%m-%d'),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature", 
            "rain", "surface_pressure", "cloud_cover", "cloud_cover_low", 
            "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration", 
            "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", 
            "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", 
            "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", 
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
            "wind_speed_10m_max", "wind_gusts_10m_max", "sunrise", "sunset", 
            "daylight_duration"
        ]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Processar dados horários
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
        "Chuva": hourly.Variables(3).ValuesAsNumpy(),
        "Pressao_Superficial": hourly.Variables(4).ValuesAsNumpy(),
        "Cobertura_Nuvens": hourly.Variables(5).ValuesAsNumpy(),
        "Cobertura_Nuvens_Baixa": hourly.Variables(6).ValuesAsNumpy(),
        "Cobertura_Nuvens_Media": hourly.Variables(7).ValuesAsNumpy(),
        "Cobertura_Nuvens_Alta": hourly.Variables(8).ValuesAsNumpy(),
        "Evapotranspiracao": hourly.Variables(9).ValuesAsNumpy(),
        "Deficit_Vapor": hourly.Variables(10).ValuesAsNumpy(),
        "Velocidade_Vento_10m": hourly.Variables(11).ValuesAsNumpy(),
        "Velocidade_Vento_100m": hourly.Variables(12).ValuesAsNumpy(),
        "Direcao_Vento_10m": hourly.Variables(13).ValuesAsNumpy(),
        "Direcao_Vento_100m": hourly.Variables(14).ValuesAsNumpy(),
        "Rajadas_Vento_10m": hourly.Variables(15).ValuesAsNumpy(),
        "Temperatura_Solo_0_7cm": hourly.Variables(16).ValuesAsNumpy(),
        "Temperatura_Solo_7_28cm": hourly.Variables(17).ValuesAsNumpy(),
        "Umidade_Solo_0_7cm": hourly.Variables(18).ValuesAsNumpy(),
        "Umidade_Solo_7_28cm": hourly.Variables(19).ValuesAsNumpy()
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Processar dados diários
    daily = response.Daily()
    daily_data = {
        "Data": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "Temperatura_Max": daily.Variables(0).ValuesAsNumpy(),
        "Temperatura_Min": daily.Variables(1).ValuesAsNumpy(),
        "Precipitacao_Total": daily.Variables(2).ValuesAsNumpy(),
        "Velocidade_Vento_10m_Max": daily.Variables(3).ValuesAsNumpy(),
        "Rajadas_Vento_10m_Max": daily.Variables(4).ValuesAsNumpy(),
        "Nascer_do_Sol": daily.Variables(5).ValuesAsNumpy(),
        "Pôr_do_Sol": daily.Variables(6).ValuesAsNumpy(),
        "Duracao_Dia": daily.Variables(7).ValuesAsNumpy()
    }
    daily_dataframe = pd.DataFrame(data=daily_data)

    return hourly_dataframe, daily_dataframe

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
        return None

# Função para obter NDVI e EVI da Embrapa
def obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final):
    access_token = obter_token_acesso_embrapa(EMBRAPA_CONSUMER_KEY, EMBRAPA_CONSUMER_SECRET)
    
    if not access_token:
        return None, None
    
    # URL da API para NDVI
    url_ndvi = 'https://api.cnptia.embrapa.br/satveg/v2/series'
    
    # Parâmetros para a requisição de NDVI
    payload_ndvi = {
        "tipoPerfil": "ndvi",
        "satelite": "comb",
        "latitude": latitude,
        "longitude": longitude,
        "dataInicial": data_inicial.strftime('%Y-%m-%d'),
        "dataFinal": data_final.strftime('%Y-%m-%d')
    }
    
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    
    # Requisição para obter NDVI
    response_ndvi = requests.post(url_ndvi, headers=headers, json=payload_ndvi)
    
    if response_ndvi.status_code == 200:
        data_ndvi = response_ndvi.json()
        df_ndvi = pd.DataFrame({
            'Data': pd.to_datetime(data_ndvi['listaDatas']),
            'NDVI': data_ndvi['listaSerie']
        })
    else:
        df_ndvi = None
        st.error(f"Erro ao obter NDVI: {response_ndvi.status_code} - {response_ndvi.json().get('user_message')
            'NDVI': data_ndvi['listaSerie']
                                                                      
        })
    else:
        df_ndvi = None
        st.error(f"Erro ao obter NDVI: {response_ndvi.status_code} - {response_ndvi.json().get('user_message', '')}")

    # Parâmetros para a requisição de EVI
    payload_evi = {
        "tipoPerfil": "evi",
        "satelite": "comb",
        "latitude": latitude,
        "longitude": longitude,
        "dataInicial": data_inicial.strftime('%Y-%m-%d'),
        "dataFinal": data_final.strftime('%Y-%m-%d')
    }
    
    # Requisição para obter EVI
    response_evi = requests.post(url_ndvi, headers=headers, json=payload_evi)
    
    if response_evi.status_code == 200:
        data_evi = response_evi.json()
        df_evi = pd.DataFrame({
            'Data': pd.to_datetime(data_evi['listaDatas']),
            'EVI': data_evi['listaSerie']
        })
    else:
        df_evi = None
        st.error(f"Erro ao obter EVI: {response_evi.status_code} - {response_evi.json().get('user_message', '')}")

    return df_ndvi, df_evi

# Função para processar e normalizar dados
def processar_dados(hourly_df, daily_df, ndvi_df, evi_df):
    # Garantir que as colunas 'Data' sejam do tipo datetime
    hourly_df['Data'] = pd.to_datetime(hourly_df['Data'])
    daily_df['Data'] = pd.to_datetime(daily_df['Data'])
    ndvi_df['Data'] = pd.to_datetime(ndvi_df['Data'])
    evi_df['Data'] = pd.to_datetime(evi_df['Data'])

    # Verificar e remover o timezone se necessário
    hourly_df['Data'] = hourly_df['Data'].dt.tz_localize(None)
    daily_df['Data'] = daily_df['Data'].dt.tz_localize(None)
    ndvi_df['Data'] = ndvi_df['Data'].dt.tz_localize(None)
    evi_df['Data'] = evi_df['Data'].dt.tz_localize(None)

    # Merge dos DataFrames
    merged_df = pd.merge(hourly_df, ndvi_df, on='Data', how='outer')
    merged_df = pd.merge(merged_df, evi_df, on='Data', how='outer')
    merged_df = pd.merge(merged_df, daily_df, on='Data', how='outer')

    # Normalização dos dados (exemplo com Min-Max)
    scaler = MinMaxScaler()
    columns_to_normalize = ['Temperatura_2m', 'Umidade_Relativa_2m', 'Temperatura_Aparente', 
                             'Chuva', 'Pressao_Superficial', 'Evapotranspiracao', 
                             'Deficit_Vapor', 'Velocidade_Vento_10m', 
                             'Velocidade_Vento_100m', 'Temperatura_Max', 
                             'Temperatura_Min', 'Precipitacao_Total', 'NDVI', 'EVI']

    # Aplicar a normalização somente nas colunas que existem no DataFrame
    for col in columns_to_normalize:
        if col in merged_df.columns:
            merged_df[col] = scaler.fit_transform(merged_df[[col]])

    return merged_df

# Função para gerar dados sintéticos para preencher NaNs
def preencher_dados_sinteticos(df):
    for column in df.columns:
        if df[column].isnull().any():  # Verifica se há NaN na coluna
            # Gerar dados sintéticos usando média da coluna (ou outra lógica)
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)  # Preencher NaNs com a média

    return df

# Definindo estados das células 
VIVO = 0
QUEIMANDO1 = 1
QUEIMANDO2 = 2
QUEIMANDO3 = 3
QUEIMANDO4 = 4
QUEIMADO = 5

# Definindo probabilidades de propagação do fogo para cada estado
def definir_probabilidades(df):
    return {
        VIVO: 0.6,
        QUEIMANDO1: 0.8 * df['Umidade_Relativa_2m'].mean(),  # Ajustar com dados da umidade
        QUEIMANDO2: 0.8,
        QUEIMANDO3: 0.8,
        QUEIMANDO4: 0.8,
        QUEIMADO: 0
    }

# Inicializando a matriz do autômato celular
def inicializar_grade(tamanho, inicio_fogo):
    grade = np.zeros((tamanho, tamanho), dtype=int)
    grade[inicio_fogo] = QUEIMANDO1
    return grade

# Calculando a probabilidade de propagação com base nos parâmetros
def calcular_probabilidade_propagacao(params):
    prob_base = 0.3  # Probabilidade base
    prob = prob_base + params['temperatura'] * 0.02  # Aumenta com a temperatura
    return min(max(prob, 0), 1)

# Aplicando a regra do autômato celular
def aplicar_regras_fogo(grade, params):
    nova_grade = grade.copy()
    tamanho = grade.shape[0]
    prob_propagacao = calcular_probabilidade_propagacao(params)

    for i in range(1, tamanho - 1):
        for j in range(1, tamanho - 1):
            if grade[i, j] == QUEIMANDO1:
                nova_grade[i, j] = QUEIMANDO2
            elif grade[i, j] == QUEIMANDO2:
                nova_grade[i, j] = QUEIMANDO3
            elif grade[i, j] == QUEIMANDO3:
                nova_grade[i, j] = QUEIMANDO4
            elif grade[i, j] == QUEIMANDO4:
                nova_grade[i, j] = QUEIMADO
                # Propaga o fogo para células adjacentes
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if grade[i + di, j + dj] == VIVO and np.random.rand() < prob_propagacao:
                        nova_grade[i + di, j + dj] = QUEIMANDO1
    return nova_grade

# Executando a simulação
def executar_simulacao(tamanho, passos, inicio_fogo, params):
    grade = inicializar_grade(tamanho, inicio_fogo)
    grades = [grade.copy()]

    for _ in range(passos):
        grade = aplicar_regras_fogo(grade, params)
        grades.append(grade.copy())

    return grades

# Plotando a simulação
def plotar_simulacao(simulacao):
    num_plots = min(50, len(simulacao))
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.flatten()
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red', 'darkred', 'black'])

    for i, grade in enumerate(simulacao[::max(1, len(simulacao)//num_plots)]):
        ax = axes[i]
        ax.imshow(grade, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Passo {i * (len(simulacao)//num_plots)}')
        ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# Interface do usuário
def main():
    st.set_page_config(page_title="Simulador de Incêndio", page_icon="🔥")

    st.title("Simulador de Propagação de Incêndio")
    
    endereco = st.text_input("Digite a localização (ex.: cidade, endereço):")
    
    # Inputs de data
    data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=7))
    data_final = st.date_input("Data Final", datetime.now())
    
    if st.button("Buscar Coordenadas"):
        latitude, longitude = obter_coordenadas_endereco(endereco)
        if latitude and longitude:
            st.session_state['latitude'] = latitude
            st.session_state['longitude'] = longitude
            st.write(f"Coordenadas encontradas: {latitude}°N, {longitude}°E")
            
            # Obter dados meteorológicos
            hourly_df, daily_df = obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final)
            if hourly_df is not None and daily_df is not None:
                st.write("### Dados Horários")
                st.dataframe(hourly_df)
                st.write("### Dados Diários")
                st.dataframe(daily_df)
            
            # Obter NDVI e EVI
            ndvi_df, evi_df = obter_ndvi_evi_embrapa(latitude, longitude, data_inicial, data_final)
            if ndvi_df is not None:
                st.write("### Dados de NDVI")
                st.dataframe(ndvi_df)
            if evi_df is not None:
                st.write("### Dados de EVI")
                st.dataframe(evi_df)
            # Processar dados
            final_df = processar_dados(hourly_df, daily_df, ndvi_df, evi_df)
            if final_df is not None:
                # Preencher dados ausentes
                final_df = preencher_dados_sinteticos(final_df)
                st.write("### Dados Processados e Normalizados")
                st.dataframe(final_df)

                # Parâmetros para a simulação
                params = {
                    'temperatura': final_df['Temperatura_2m'].mean(),
                    'umidade': final_df['Umidade_Relativa_2m'].mean(),
                    'velocidade_vento': final_df['Velocidade_Vento_10m'].mean(),
                    'densidade_vegetacao': final_df['NDVI'].mean() * 100,  # Normalizando para 0-100
                    'umidade_combustivel': final_df['Umidade_Solo_0_7cm'].mean(),
                    'topografia': 10,  # Pode ser ajustado ou coletado de outros dados
                    'direcao_vento': 90,  # Exemplo fixo, pode ser variável
                    'intensidade_fogo': 10000,  # Exemplo fixo, pode ser variável
                    'tipo_vegetacao': 'floresta tropical',  # Exemplo fixo
                    'tipo_solo': 'argiloso',  # Exemplo fixo
                    'intervencao_humana': 0.1  # Exemplo fixo
                }

                # Configurações da simulação
                tamanho = 50  # Tamanho da grade
                passos = 10  # Número de passos na simulação
                inicio_fogo = (25, 25)  # Posição inicial do fogo

                # Executar a simulação
                simulacao = executar_simulacao(tamanho, passos, inicio_fogo, params)

                # Plotar a simulação
                st.write("### Simulação de Incêndio")
                plotar_simulacao(simulacao)

if __name__ == "__main__":
    main()

