import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import base64
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
from fpdf import FPDF

# Definições iniciais e configurações globais
st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

# Estados das células
VIVO = 0
QUEIMANDO = 1
QUEIMADO = 2

# Cores para visualização
colors = {
    VIVO: 'green',
    QUEIMANDO: 'red',
    QUEIMADO: 'black'
}

# Credenciais das APIs
consumer_key = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
consumer_secret = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Funções para integração com APIs
def obter_token_acesso(consumer_key, consumer_secret):
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
        return lista_ndvi_evi, lista_datas
    else:
        st.error(f"Erro ao obter dados NDVI/EVI: {response.status_code} - {response.text}")
        return None, None

def obter_dados_climaticos(latitude, longitude):
    access_token = obter_token_acesso(consumer_key, consumer_secret)
    if access_token is None:
        return None
    variaveis = {
        'temperatura': 'tmpsfc',
        'umidade': 'rh2m',
        'velocidade_vento': ['ugrd10m', 'vgrd10m'],
        'precipitacao': 'apcpsfc',
        'radiacao_solar': 'sunsdsfc'
    }
    dados_climaticos = {}
    for param, var_api in variaveis.items():
        if isinstance(var_api, list):
            # Componentes U e V do vento
            data_execucao = obter_ultima_data_execucao(var_api[0], access_token)
            previsao_u = obter_previsao(var_api[0], data_execucao, latitude, longitude, access_token)
            previsao_v = obter_previsao(var_api[1], data_execucao, latitude, longitude, access_token)
            if previsao_u is None or previsao_v is None:
                continue
            u = previsao_u[0]['valor']
            v = previsao_v[0]['valor']
            velocidade_vento = (u**2 + v**2)**0.5  # Velocidade do vento
            direcao_vento = (np.arctan2(v, u) * 180 / np.pi) % 360  # Direção do vento em graus
            dados_climaticos['velocidade_vento'] = velocidade_vento
            dados_climaticos['direcao_vento'] = direcao_vento
        else:
            data_execucao = obter_ultima_data_execucao(var_api, access_token)
            previsao = obter_previsao(var_api, data_execucao, latitude, longitude, access_token)
            if previsao is None:
                continue
            valor = previsao[0]['valor']
            if param == 'temperatura':
                valor -= 273.15  # Converter de Kelvin para Celsius
            dados_climaticos[param] = valor
    return dados_climaticos

def obter_ultima_data_execucao(variavel, access_token):
    url = f'https://api.cnptia.embrapa.br/climapi/v1/ncep-gfs/{variavel}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        datas = response.json()
        if datas:
            return datas[-1]
    return None

def obter_previsao(variavel, data_execucao, latitude, longitude, access_token):
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

def classificar_solo(dados_perfil):
    access_token = obter_token_acesso(consumer_key, consumer_secret)
    if access_token is None:
        return None
    url = 'https://api.cnptia.embrapa.br/sibcs/v1/classification'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=dados_perfil)

    if response.status_code == 200:
        classificacao = response.json()
        return classificacao
    else:
        st.error(f"Erro ao classificar o solo: {response.status_code} - {response.text}")
        return None

# Funções auxiliares
def coordenadas_validas(latitude, longitude):
    return -60.0 <= latitude <= 15.0 and -90.0 <= longitude <= -30.0

def gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula):
    # Criar um GeoDataFrame para visualização
    polygons = []
    states = []
    tamanho_grade = simulacao[0].shape[0]
    for i in range(tamanho_grade):
        for j in range(tamanho_grade):
            x = longitude + (j - tamanho_grade / 2) * tamanho_celula
            y = latitude + (i - tamanho_grade / 2) * tamanho_celula
            polygon = Polygon([
                (x, y),
                (x + tamanho_celula, y),
                (x + tamanho_celula, y + tamanho_celula),
                (x, y + tamanho_celula)
            ])
            state = simulacao[-1][i, j]
            polygons.append(polygon)
            states.append(state)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'state': states})
    # Mapa interativo
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            'fillColor': colors.get(feature['properties']['state'], 'gray'),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        }
    ).add_to(m)
    st_folium(m, width=700, height=500)

def gerar_serie_historica(lista_datas, lista_ndvi_evi):
    df = pd.DataFrame({'Data': pd.to_datetime(lista_datas), 'Índice': lista_ndvi_evi})
    fig = px.line(df, x='Data', y='Índice', title='Série Histórica de NDVI/EVI')
    st.plotly_chart(fig)

def executar_simulacao(params, tamanho_grade, num_passos):
    # Inicializar a grade
    grade = np.full((tamanho_grade, tamanho_grade), VIVO)
    # Definir ponto de ignição
    centro = tamanho_grade // 2
    grade[centro, centro] = QUEIMANDO
    grades = [grade.copy()]

    # Executar simulação
    for passo in range(num_passos):
        nova_grade = grade.copy()
        for i in range(tamanho_grade):
            for j in range(tamanho_grade):
                if grade[i, j] == QUEIMANDO:
                    nova_grade[i, j] = QUEIMADO
                    # Propagar para vizinhos
                    vizinhos = [(-1,0), (1,0), (0,-1), (0,1)]
                    for vi, vj in vizinhos:
                        ni, nj = i + vi, j + vj
                        if 0 <= ni < tamanho_grade and 0 <= nj < tamanho_grade:
                            if grade[ni, nj] == VIVO:
                                probabilidade = calcular_probabilidade_propagacao(params)
                                if np.random.rand() < probabilidade:
                                    nova_grade[ni, nj] = QUEIMANDO
        grade = nova_grade
        grades.append(grade.copy())
    return grades

def calcular_probabilidade_propagacao(params):
    # Simples combinação de fatores
    prob = params['fator_combustivel'] * params['fator_climatico'] * params['fator_terreno']
    return min(max(prob, 0), 1)

def gerar_relatorio_pdf(resultados):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Relatório de Simulação de Propagação de Incêndio", ln=True, align='C')

    pdf.cell(200, 10, txt="Parâmetros Utilizados:", ln=True)
    for key, value in resultados['params'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.cell(200, 10, txt="Resultados:", ln=True)
    for key, value in resultados['resultados'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

# Interface do usuário
def main():
    st.title("EcoSim.ai")
    st.subheader("Simulador Inovador de Propagação de Incêndio")

    # Seção para seleção de localização
    st.header("Seleção de Localização")
    m = folium.Map(location=[-15.793889, -47.882778], zoom_start=4)
    map_data = st_folium(m, width=700, height=500)

    if map_data['last_clicked'] is not None:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['lng']
        st.write(f"Latitude: {latitude:.6f}")
        st.write(f"Longitude: {longitude:.6f}")
    else:
        st.warning("Por favor, clique no mapa para selecionar a localização.")
        return

    if not coordenadas_validas(latitude, longitude):
        st.error("As coordenadas selecionadas não estão dentro dos limites da América do Sul.")
        return

    # Obter dados das APIs
    st.header("Obtenção de Dados das APIs")
    tipo_indice = st.selectbox('Tipo de Índice Vegetativo', ['ndvi', 'evi'])
    satelite = st.selectbox('Satélite', ['terra', 'aqua', 'comb'])

    if st.button('Obter NDVI/EVI da API'):
        lista_ndvi_evi, lista_datas = obter_ndvi_evi(latitude, longitude, tipo_indice, satelite)
        if lista_ndvi_evi is not None:
            ndvi_evi_atual = lista_ndvi_evi[-1]
            st.success(f"Valor atual de {tipo_indice.upper()}: {ndvi_evi_atual}")
            gerar_serie_historica(lista_datas, lista_ndvi_evi)
        else:
            st.error("Não foi possível obter o NDVI/EVI.")
            return
    else:
        ndvi_evi_atual = 0.5  # Valor padrão se não obtido

    if st.button('Obter Dados Climáticos da API'):
        dados_climaticos = obter_dados_climaticos(latitude, longitude)
        if dados_climaticos is not None:
            st.success("Dados climáticos obtidos com sucesso!")
            st.write(dados_climaticos)
        else:
            st.error("Não foi possível obter os dados climáticos.")
            return
    else:
        dados_climaticos = {
            'temperatura': 25,
            'umidade': 50,
            'velocidade_vento': 10,
            'direcao_vento': 90,
            'precipitacao': 0,
            'radiacao_solar': 800
        }

    # Entrada de dados manuais
    st.header("Dados Manuais")
    st.write("Caso deseje, você pode ajustar os parâmetros manualmente.")

    temperatura = st.slider('Temperatura (°C)', -10, 50, int(dados_climaticos.get('temperatura', 25)))
    umidade = st.slider('Umidade Relativa (%)', 0, 100, int(dados_climaticos.get('umidade', 50)))
    velocidade_vento = st.slider('Velocidade do Vento (km/h)', 0, 100, int(dados_climaticos.get('velocidade_vento', 10)))
    direcao_vento = st.slider('Direção do Vento (graus)', 0, 360, int(dados_climaticos.get('direcao_vento', 90)))
    precipitacao = st.slider('Precipitação (mm)', 0, 200, int(dados_climaticos.get('precipitacao', 0)))
    radiacao_solar = st.slider('Radiação Solar (W/m²)', 0, 1200, int(dados_climaticos.get('radiacao_solar', 800)))
    ndvi = st.slider('NDVI', 0.0, 1.0, float(ndvi_evi_atual))

    # Classificação do solo
    st.header("Classificação do Solo")
    if st.button('Classificar Solo com a API'):
        # Dados do perfil de solo (simplificado para o exemplo)
        dados_perfil = {
            "items": [
                {
                    "ID_PONTO": "Ponto1",
                    "DRENAGEM": 1,
                    "HORIZONTES": [
                        {
                            "SIMB_HORIZ": "A",
                            "LIMITE_SUP": 0,
                            "LIMITE_INF": 20,
                            "AREIA_GROS": 15.0,
                            "AREIA_FINA": 25.0,
                            "SILTE": 30.0,
                            "ARGILA": 30.0,
                            "PH_AGUA": 6.0,
                            "C_ORG": 2.5,
                            "CA_TROC": 5.0,
                            "MG_TROC": 1.5,
                            "K_TROC": 0.2,
                            "NA_TROC": 0.1,
                            "AL_TROC": 0.0,
                            "H_TROC": 4.0,
                            "P_ASSIM": 15.0,
                            "RETRATIL": False,
                            "COESO": False
                        }
                    ]
                }
            ]
        }
        classificacao = classificar_solo(dados_perfil)
        if classificacao is not None:
            st.success(f"Classificação do Solo: {classificacao['items'][0]['ORDEM']}")
            tipo_solo = classificacao['items'][0]['ORDEM']
        else:
            st.error("Não foi possível classificar o solo.")
            tipo_solo = 'Desconhecido'
    else:
        tipo_solo = 'Desconhecido'

    # Parâmetros para a simulação
    params = {
        'temperatura': temperatura,
        'umidade': umidade,
        'velocidade_vento': velocidade_vento,
        'direcao_vento': direcao_vento,
        'precipitacao': precipitacao,
        'radiacao_solar': radiacao_solar,
        'ndvi': ndvi,
        'tipo_solo': tipo_solo,
        'fator_combustivel': ndvi,  # Exemplo simplificado
        'fator_climatico': (temperatura / 40) * ((100 - umidade) / 100),
        'fator_terreno': 1  # Pode incluir outros fatores
    }

    # Execução da simulação
    if st.button('Executar Simulação'):
        tamanho_grade = 50  # Pode ajustar conforme necessário
        num_passos = 100
        simulacao = executar_simulacao(params, tamanho_grade, num_passos)
        st.success("Simulação concluída!")
        # Visualização da propagação
        gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula=0.01)
        # Gerar relatório
        resultados = {
            'params': params,
            'resultados': {
                'Área Queimada (km²)': np.sum(simulacao[-1] == QUEIMADO) * (0.01 ** 2)
            }
        }
        pdf_bytes = gerar_relatorio_pdf(resultados)
        st.download_button(label="Baixar Relatório em PDF", data=pdf_bytes, file_name="relatorio_simulacao.pdf", mime="application/pdf")
    else:
        st.info("Ajuste os parâmetros e clique em 'Executar Simulação'.")

if __name__ == '__main__':
    main()
