import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import base64
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, Point
from fpdf import FPDF
import urllib.parse  # Para codificar o endereço na URL
import datetime  # Para manipulação de datas

# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="EcoSim.ai - Simulador de Propagação de Incêndio",
    page_icon="🔥",
    layout="wide"
)

# Definição dos estados das células na simulação
VIVO = 0        # Vegetação não queimada
QUEIMANDO = 1   # Vegetação em chamas
QUEIMADO = 2    # Vegetação já queimada

# Cores associadas a cada estado para visualização no mapa
colors = {
    VIVO: 'green',
    QUEIMANDO: 'red',
    QUEIMADO: 'black'
}

# Credenciais para acesso às APIs (mantenha essas informações seguras)
consumer_key = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
consumer_secret = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Funções para integração com as APIs
def obter_token_acesso(consumer_key, consumer_secret):
    """
    Obtém o token de acesso para autenticação nas APIs da Embrapa.
    """
    token_url = 'https://api.cnptia.embrapa.br/token'
    credentials = f"{consumer_key}:{consumer_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'client_credentials'
    }

    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info['access_token']
        return access_token
    else:
        st.error(f"Erro ao obter token de acesso: {response.status_code} - {response.text}")
        return None

def obter_ndvi_evi(latitude, longitude, tipo_indice='ndvi', satelite='comb'):
    """
    Obtém a série temporal de NDVI ou EVI para a localização especificada.
    """
    access_token = obter_token_acesso(consumer_key, consumer_secret)
    if access_token is None:
        return None, None
    url = 'https://api.cnptia.embrapa.br/satveg/v2/series'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    payload = {
        "tipoPerfil": tipo_indice,
        "satelite": satelite,
        "latitude": latitude,
        "longitude": longitude
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        lista_ndvi_evi = data['listaSerie']
        lista_datas = data['listaDatas']
        # Corrigir a análise de datas de acordo com o formato retornado pela API
        try:
            # Ajustar o formato da data com base no formato real retornado pela API
            lista_datas = [datetime.strptime(data_str, '%Y-%m-%d') for data_str in lista_datas]
        except ValueError:
            try:
                lista_datas = [datetime.strptime(data_str, '%d/%m/%Y') for data_str in lista_datas]
            except ValueError:
                st.error("Formato de data não reconhecido. Verifique o formato de data retornado pela API.")
                return None, None
        return lista_ndvi_evi, lista_datas
    else:
        st.error(f"Erro ao obter dados NDVI/EVI: {response.status_code} - {response.text}")
        return None, None

def obter_dados_climaticos(latitude, longitude, data_consulta, access_token):
    """
    Obtém as séries temporais dos dados climáticos para a data de execução especificada e localização.
    """
    variaveis = {
        'temperatura': 'tmpsfc',
        'umidade': 'rh2m',
        'vento_u': 'ugrd10m',
        'vento_v': 'vgrd10m',
        'precipitacao': 'apcpsfc',
        'radiacao_solar': 'sunsdsfc'
    }
    series_temporais = {}
    for param, var_api in variaveis.items():
        data_execucao = data_consulta  # Usar a data selecionada
        previsao = obter_previsao(var_api, data_execucao, latitude, longitude, access_token)
        if previsao is None:
            st.error(f"Não foi possível obter dados para {param}.")
            continue
        tempos = []
        valores = []
        for ponto in previsao:
            if 'time' in ponto:
                tempos.append(datetime.datetime.fromtimestamp(ponto['time'] / 1000))  # Usar 'time' em vez de 'data'
            else:
                st.error(f"Chave 'time' não encontrada na resposta da API para {param}.")
                continue
            if 'valor' in ponto:
                valor = ponto['valor']
            else:
                st.error(f"Chave 'valor' não encontrada na resposta da API para {param}.")
                continue
            if param == 'temperatura':
                valor -= 273.15  # Converter de Kelvin para Celsius
            valores.append(valor)
        series_temporais[param] = {'tempos': tempos, 'valores': valores}
    # Processar velocidade e direção do vento
    if 'vento_u' in series_temporais and 'vento_v' in series_temporais:
        tempos = series_temporais['vento_u']['tempos']
        u_valores = np.array(series_temporais['vento_u']['valores'])
        v_valores = np.array(series_temporais['vento_v']['valores'])
        velocidade_vento = np.sqrt(u_valores ** 2 + v_valores ** 2)
        direcao_vento = (np.arctan2(v_valores, u_valores) * 180 / np.pi) % 360
        series_temporais['velocidade_vento'] = {'tempos': tempos, 'valores': velocidade_vento.tolist()}
        series_temporais['direcao_vento'] = {'tempos': tempos, 'valores': direcao_vento.tolist()}
        del series_temporais['vento_u']
        del series_temporais['vento_v']
    return series_temporais

def obter_previsao(variavel, data_execucao, latitude, longitude, access_token):
    """
    Obtém a previsão da variável climática especificada para a data de execução, latitude e longitude.
    """
    url = f'https://api.cnptia.embrapa.br/climapi/v1/ncep-gfs/{variavel}/{data_execucao}/{longitude}/{latitude}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        previsao = response.json()
        return previsao
    else:
        return None

def obter_dados_climaticos_openweathermap(latitude, longitude, data_consulta, api_key):
    """
    Obtém dados climáticos históricos do OpenWeatherMap para a data e localização especificadas.
    """
    timestamp = int(datetime.datetime.strptime(data_consulta, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        'lat': latitude,
        'lon': longitude,
        'dt': timestamp,
        'appid': api_key,
        'units': 'metric',
        'lang': 'pt'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        series_temporais = {}
        tempos = [datetime.datetime.fromtimestamp(hourly_data['dt']) for hourly_data in data['hourly']]
        # Temperatura
        temperaturas = [hourly_data['temp'] for hourly_data in data['hourly']]
        series_temporais['temperatura'] = {'tempos': tempos, 'valores': temperaturas}
        # Umidade
        umidades = [hourly_data['humidity'] for hourly_data in data['hourly']]
        series_temporais['umidade'] = {'tempos': tempos, 'valores': umidades}
        # Velocidade do Vento
        velocidades_vento = [hourly_data['wind_speed'] for hourly_data in data['hourly']]
        series_temporais['velocidade_vento'] = {'tempos': tempos, 'valores': velocidades_vento}
        # Direção do Vento
        direcoes_vento = [hourly_data.get('wind_deg', 0) for hourly_data in data['hourly']]
        series_temporais['direcao_vento'] = {'tempos': tempos, 'valores': direcoes_vento}
        # Precipitação
        precipitacoes = [hourly_data.get('rain', {}).get('1h', 0) for hourly_data in data['hourly']]
        series_temporais['precipitacao'] = {'tempos': tempos, 'valores': precipitacoes}
        # Radiação Solar (não disponível diretamente, usar zeros ou estimar)
        radiacao_solar = [0]*len(tempos)  # Placeholder
        series_temporais['radiacao_solar'] = {'tempos': tempos, 'valores': radiacao_solar}
        return series_temporais
    else:
        st.error(f"Erro ao obter dados climáticos do OpenWeatherMap: {response.status_code}")
        return None

# [As demais funções permanecem as mesmas...]

# Interface do usuário
def main():
    """
    Função principal que controla a interface do usuário e a lógica do aplicativo.
    """
    st.title("EcoSim.ai")
    st.subheader("Simulador Inovador de Propagação de Incêndio")

    # Seção para entrada de cidade ou endereço
    st.header("Entrada de Cidade ou Endereço")
    endereco = st.text_input("Digite o nome da cidade ou endereço:")
    if st.button("Obter Coordenadas"):
        if endereco:
            latitude, longitude = obter_coordenadas_endereco(endereco)
            if latitude and longitude:
                st.success(f"Coordenadas obtidas: Latitude {latitude:.6f}, Longitude {longitude:.6f}")
                st.session_state.latitude = latitude
                st.session_state.longitude = longitude
            else:
                st.error("Não foi possível obter as coordenadas para o endereço fornecido.")
                return
        else:
            st.error("Por favor, insira um endereço válido.")
            return

    # Ou selecionar no mapa
    st.header("Ou clique no mapa para selecionar a localização")
    m = folium.Map(location=[-15.793889, -47.882778], zoom_start=4)
    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        folium.Marker([st.session_state.latitude, st.session_state.longitude], tooltip="Localização Selecionada").add_to(m)
    map_data = st_folium(m, width=700, height=500)

    if map_data['last_clicked'] is not None:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['lng']
        st.success(f"Coordenadas selecionadas: Latitude {latitude:.6f}, Longitude {longitude:.6f}")
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude

    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.warning("Por favor, insira um endereço ou clique no mapa para selecionar a localização.")
        return
    else:
        latitude = st.session_state.latitude
        longitude = st.session_state.longitude

    if not coordenadas_validas(latitude, longitude):
        st.error("As coordenadas selecionadas não estão dentro dos limites da América do Sul.")
        return

    # Obter dados das APIs
    st.header("Obtenção de Dados das APIs")
    tipo_indice = st.selectbox('Tipo de Índice Vegetativo', ['ndvi', 'evi'])
    satelite = st.selectbox('Satélite', ['terra', 'aqua', 'comb'])

    # Obter NDVI/EVI
    if st.button('Obter NDVI/EVI da API'):
        lista_ndvi_evi, lista_datas = obter_ndvi_evi(latitude, longitude, tipo_indice, satelite)
        if lista_ndvi_evi is not None:
            ndvi_evi_atual = lista_ndvi_evi[-1]
            st.success(f"Valor atual de {tipo_indice.upper()}: {ndvi_evi_atual}")
            gerar_serie_historica(lista_datas, lista_ndvi_evi)
            st.session_state.ndvi_evi_atual = ndvi_evi_atual
            st.session_state.lista_ndvi_evi = lista_ndvi_evi
            st.session_state.lista_datas_ndvi_evi = lista_datas
            ndvi = ndvi_evi_atual
        else:
            st.error("Não foi possível obter o NDVI/EVI.")
            return
    else:
        ndvi_evi_atual = st.session_state.get('ndvi_evi_atual', 0.5)
        ndvi = ndvi_evi_atual
        lista_ndvi_evi = st.session_state.get('lista_ndvi_evi', [])
        lista_datas_ndvi_evi = st.session_state.get('lista_datas_ndvi_evi', [])
        if lista_ndvi_evi and lista_datas_ndvi_evi:
            gerar_serie_historica(lista_datas_ndvi_evi, lista_ndvi_evi)

    # Obter dados climáticos
    st.header("Obtenção de Dados Climáticos")

    # Solicitar data ao usuário
    data_consulta = st.date_input("Selecione a data para os dados climáticos:", datetime.datetime.now().date())

    # Solicitar chave de API do OpenWeatherMap
    api_key_openweathermap = st.text_input("Insira sua chave de API do OpenWeatherMap:", type="password")

    if st.button('Obter Dados Climáticos das APIs'):
        data_consulta_str = data_consulta.strftime('%Y-%m-%d')
        # Tentar obter dados da API da Embrapa
        access_token = obter_token_acesso(consumer_key, consumer_secret)
        series_temporais = None
        if access_token:
            series_temporais = obter_dados_climaticos(latitude, longitude, data_consulta_str, access_token)
        
        # Se não conseguiu obter dados da Embrapa, tentar OpenWeatherMap
        if not series_temporais or not series_temporais.get('temperatura'):
            if api_key_openweathermap:
                series_temporais = obter_dados_climaticos_openweathermap(latitude, longitude, data_consulta_str, api_key_openweathermap)
            else:
                st.error("Por favor, insira sua chave de API do OpenWeatherMap.")
                return

        if series_temporais:
            st.success("Dados climáticos obtidos com sucesso!")
            st.session_state.series_temporais = series_temporais
            # Atualizar os valores dos parâmetros manuais com os dados mais recentes
            temperatura = series_temporais['temperatura']['valores'][-1]
            umidade = series_temporais['umidade']['valores'][-1]
            velocidade_vento = series_temporais['velocidade_vento']['valores'][-1]
            direcao_vento = series_temporais['direcao_vento']['valores'][-1]
            precipitacao = series_temporais['precipitacao']['valores'][-1]
            radiacao_solar = series_temporais['radiacao_solar']['valores'][-1]
            # Exibir tabelas e gráficos
            st.header("Séries Temporais Climáticas")
            for param, data in series_temporais.items():
                df = pd.DataFrame({'Data': data['tempos'], 'Valor': data['valores']})
                st.subheader(f"{param.capitalize()}")
                st.dataframe(df)
                fig = px.line(df, x='Data', y='Valor', title=f'Série Temporal de {param.capitalize()}')
                st.plotly_chart(fig)
        else:
            st.error("Não foi possível obter os dados climáticos de nenhuma API.")
            return
    else:
        series_temporais = st.session_state.get('series_temporais', None)
        if series_temporais:
            temperatura = series_temporais['temperatura']['valores'][-1]
            umidade = series_temporais['umidade']['valores'][-1]
            velocidade_vento = series_temporais['velocidade_vento']['valores'][-1]
            direcao_vento = series_temporais['direcao_vento']['valores'][-1]
            precipitacao = series_temporais['precipitacao']['valores'][-1]
            radiacao_solar = series_temporais['radiacao_solar']['valores'][-1]
        else:
            temperatura = 25
            umidade = 50
            velocidade_vento = 10
            direcao_vento = 90
            precipitacao = 0
            radiacao_solar = 800

    # [As demais partes do código permanecem as mesmas, incluindo a seção de ajuste de parâmetros, execução da simulação, etc.]

if __name__ == '__main__':
    main()
