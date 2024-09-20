import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import base64
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon
from fpdf import FPDF
import urllib.parse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Chaves de API (substitua pelas suas próprias)

API_KEY_OPENWEATHERMAP = '5af75d4ed5ae582e673f8d0ba4728936'
EMBRAPA_CONSUMER_KEY = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
EMBRAPA_CONSUMER_SECRET = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Funções para integração com as APIs
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

def obter_dados_meteorologicos(latitude, longitude, data_inicial, data_final):
    api_url = 'https://api.openweathermap.org/data/2.5/onecall/timemachine'
    # Converter datas para datetime.datetime
    datas = [datetime.combine(data_final - timedelta(days=i), datetime.min.time()) for i in range((data_final - data_inicial).days + 1)]
    series = []
    for data in datas:
        params = {
            'lat': latitude,
            'lon': longitude,
            'dt': int(data.timestamp()),
            'appid': API_KEY_OPENWEATHERMAP,
            'units': 'metric',
            'lang': 'pt'
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data_json = response.json()
            for hour_data in data_json.get('hourly', []):
                series.append({
                    'Data': datetime.fromtimestamp(hour_data['dt']),
                    'Temperatura': hour_data['temp'],
                    'Umidade': hour_data['humidity'],
                    'Vento': hour_data['wind_speed']
                })
        else:
            st.warning(f"Erro ao obter dados meteorológicos: {response.status_code}")
    if series:
        return pd.DataFrame(series)
    else:
        return None

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

def coordenadas_validas(latitude, longitude):
    return -60.0 <= latitude <= 15.0 and -90.0 <= longitude <= -30.0

# Funções de simulação
def calcular_probabilidade_propagacao(parametros):
    temperatura = parametros['temperatura']
    umidade = parametros['umidade']
    vento = parametros['vento']

    prob = (temperatura / 40) * ((100 - umidade) / 100) * (vento / 10)
    return min(max(prob, 0), 1)

def simular_propagacao_incendio(grade_inicial, parametros, num_passos):
    tamanho = grade_inicial.shape[0]
    grades = [grade_inicial.copy()]
    for passo in range(num_passos):
        nova_grade = grades[-1].copy()
        for i in range(tamanho):
            for j in range(tamanho):
                estado_atual = grades[-1][i, j]
                if estado_atual == QUEIMANDO:
                    nova_grade[i, j] = QUEIMADO
                    vizinhos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for vi, vj in vizinhos:
                        ni, nj = i + vi, j + vj
                        if 0 <= ni < tamanho and 0 <= nj < tamanho:
                            if grades[-1][ni, nj] == VIVO:
                                prob = calcular_probabilidade_propagacao(parametros)
                                if np.random.rand() < prob:
                                    nova_grade[ni, nj] = QUEIMANDO
                elif estado_atual == QUEIMADO:
                    prob_recuperacao = parametros['prob_recuperacao']
                    if np.random.rand() < prob_recuperacao:
                        nova_grade[i, j] = RECUPERADO
        grades.append(nova_grade)
    return grades

def criar_animacao_simulacao(grades):
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('brg', 4)
    im = ax.imshow(grades[0], cmap=cmap, vmin=VIVO, vmax=RECUPERADO)
    plt.colorbar(im, ticks=range(4), label='Estado')

    def update(frame):
        im.set_data(grades[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(grades), interval=200, blit=True)
    return ani

def gerar_mapa_propagacao(grade_final, latitude, longitude, tamanho_celula):
    tamanho_grade = grade_final.shape[0]
    polygons = []
    states = []
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
            state = grade_final[i, j]
            polygons.append(polygon)
            states.append(state)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'state': states}, crs='EPSG:4326')

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

def gerar_relatorio(parametros, resultados):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Simulação de Incêndio", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt="Parâmetros Utilizados:", ln=True)
    for key, value in parametros.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt="Resultados:", ln=True)
    for key, value in resultados.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_output = pdf.output(dest='S').encode('latin1')
    st.download_button("Baixar Relatório em PDF", data=pdf_output, file_name="relatorio_simulacao.pdf", mime='application/pdf')

# Interface do usuário
def main():
    st.title("Simulador de Propagação de Incêndio")
    st.subheader("Automação de Parâmetros Usando APIs")

    # Entrada de localização
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

    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        lat = st.session_state['latitude']
        lon = st.session_state['longitude']

        if not coordenadas_validas(lat, lon):
            st.error("Coordenadas fora dos limites permitidos.")
            return

        # Período de análise
        st.header("Período de Análise")
        data_inicial = st.date_input("Data Inicial", datetime.now() - timedelta(days=30))
        data_final = st.date_input("Data Final", datetime.now())

        if data_final < data_inicial:
            st.error("A data final deve ser posterior à data inicial.")
            return

        # Obter NDVI/EVI
        st.header("Dados de Vegetação")
        tipo_indice = st.selectbox("Selecione o Índice", ["ndvi", "evi"])
        if st.button("Obter NDVI/EVI"):
            ndvi_series = obter_ndvi_evi_embrapa(lat, lon, data_inicial, data_final, tipo_indice=tipo_indice)
            if ndvi_series is not None:
                st.line_chart(ndvi_series.set_index('Data'))

        # Obter dados meteorológicos
        st.header("Dados Meteorológicos")
        if st.button("Obter Dados Meteorológicos"):
            meteo_series = obter_dados_meteorologicos(lat, lon, data_inicial, data_final)
            if meteo_series is not None:
                st.line_chart(meteo_series.set_index('Data'))

        # Parâmetros da simulação
        st.header("Parâmetros da Simulação")
        num_passos = st.slider("Número de Passos da Simulação", min_value=10, max_value=200, value=50)
        tamanho_grade = st.slider("Tamanho da Grade", min_value=10, max_value=100, value=50)
        prob_recuperacao = st.slider("Probabilidade de Recuperação da Vegetação", min_value=0.0, max_value=1.0, value=0.1)

        if st.button("Executar Simulação"):
            # Configurar parâmetros com base nos dados obtidos
            parametros = {
                'temperatura': meteo_series['Temperatura'].mean() if meteo_series is not None else 25,
                'umidade': meteo_series['Umidade'].mean() if meteo_series is not None else 50,
                'vento': meteo_series['Vento'].mean() if meteo_series is not None else 5,
                'prob_recuperacao': prob_recuperacao
            }

            # Inicializar grade
            grade_inicial = np.full((tamanho_grade, tamanho_grade), VIVO)
            centro = tamanho_grade // 2
            grade_inicial[centro, centro] = QUEIMANDO

            # Executar simulação
            grades = simular_propagacao_incendio(grade_inicial, parametros, num_passos)

            # Exibição dos resultados
            st.header("Resultado da Simulação")

            # Mapa de propagação final
            st.subheader("Mapa de Propagação Final")
            gerar_mapa_propagacao(grades[-1], lat, lon, tamanho_celula=0.001)

            # Animação (opcional)
            st.subheader("Animação da Simulação")
            animacao = criar_animacao_simulacao(grades)
            st.pyplot(animacao)

            # Geração de relatório
            resultados = {
                'Área Queimada (unidades)': np.sum(grades[-1] == QUEIMADO),
                'Vegetação Recuperada (unidades)': np.sum(grades[-1] == RECUPERADO)
            }
            st.write(resultados)
            gerar_relatorio(parametros, resultados)

    else:
        st.info("Por favor, insira uma localização e obtenha as coordenadas.")

if __name__ == '__main__':
    main()

