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
            lista_datas = [datetime.datetime.strptime(data_str, '%Y-%m-%d') for data_str in lista_datas]
        except ValueError:
            try:
                lista_datas = [datetime.datetime.strptime(data_str, '%d/%m/%Y') for data_str in lista_datas]
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
            st.warning(f"Não foi possível obter dados para {param}. Tentando próxima API.")
            return None
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

def classificar_solo(dados_perfil):
    """
    Classifica o solo com base nos dados do perfil fornecidos.
    """
    access_token = obter_token_acesso(consumer_key, consumer_secret)
    if access_token is None:
        return None
    url = 'https://api.cnptia.embrapa.br/sibcs/classification'
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
    """
    Verifica se as coordenadas estão dentro dos limites da América do Sul.
    """
    return -60.0 <= latitude <= 15.0 and -90.0 <= longitude <= -30.0

def obter_coordenadas_endereco(endereco):
    """
    Obtém as coordenadas (latitude e longitude) para o endereço fornecido usando a API Nominatim.
    """
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(endereco)}&format=json&limit=1"
    headers = {
        'User-Agent': 'EcoSim.ai/1.0 (contato@exemplo.com)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        resultado = response.json()
        if resultado:
            latitude = float(resultado[0]['lat'])
            longitude = float(resultado[0]['lon'])
            return latitude, longitude
        else:
            st.error("Endereço não encontrado.")
            return None, None
    else:
        st.error(f"Erro ao consultar o serviço de geocodificação. Código de status: {response.status_code}")
        return None, None

def gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula):
    """
    Gera o mapa de propagação do incêndio com base na simulação.
    """
    polygons = []
    states = []
    tamanho_grade = simulacao[0].shape[0]
    for i in range(tamanho_grade):
        for j in range(tamanho_grade):
            # Calcular as coordenadas de cada célula
            x = longitude + (j - tamanho_grade / 2) * tamanho_celula
            y = latitude + (i - tamanho_grade / 2) * tamanho_celula
            # Criar um polígono para cada célula
            polygon = Polygon([
                (x, y),
                (x + tamanho_celula, y),
                (x + tamanho_celula, y + tamanho_celula),
                (x, y + tamanho_celula)
            ])
            # Obter o estado da célula (VIVO, QUEIMANDO, QUEIMADO)
            state = simulacao[-1][i, j]
            polygons.append(polygon)
            states.append(state)

    # Criar um GeoDataFrame com os polígonos e estados
    gdf = gpd.GeoDataFrame({'geometry': polygons, 'state': states}, crs='EPSG:4326')

    # Criar o mapa interativo com o Folium
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
    """
    Gera um gráfico de série histórica para o NDVI/EVI.
    """
    df = pd.DataFrame({'Data': pd.to_datetime(lista_datas), 'Índice': lista_ndvi_evi})
    fig = px.line(df, x='Data', y='Índice', title='Série Histórica de NDVI/EVI')
    st.plotly_chart(fig)

def executar_simulacao(params, tamanho_grade, num_passos):
    """
    Executa a simulação de propagação de incêndio com base nos parâmetros fornecidos.
    """
    # Inicializar a grade com células VIVAS
    grade = np.full((tamanho_grade, tamanho_grade), VIVO)

    # Definir ponto de ignição no centro da grade
    centro = tamanho_grade // 2
    grade[centro, centro] = QUEIMANDO

    # Lista para armazenar as grades em cada passo de tempo
    grades = [grade.copy()]

    # Executar a simulação por 'num_passos' passos de tempo
    for passo in range(num_passos):
        nova_grade = grade.copy()
        for i in range(tamanho_grade):
            for j in range(tamanho_grade):
                if grade[i, j] == QUEIMANDO:
                    # Célula que está queimando agora ficará QUEIMADA no próximo passo
                    nova_grade[i, j] = QUEIMADO
                    # Propagar o fogo para os vizinhos
                    vizinhos = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Vizinhos ortogonais
                    for vi, vj in vizinhos:
                        ni, nj = i + vi, j + vj
                        # Verificar se o vizinho está dentro dos limites da grade
                        if 0 <= ni < tamanho_grade and 0 <= nj < tamanho_grade:
                            # Propagar somente para células VIVAS
                            if grade[ni, nj] == VIVO:
                                # Calcular a probabilidade de propagação
                                probabilidade = calcular_probabilidade_propagacao(params)
                                # Decidir se o fogo irá propagar para a célula vizinha
                                if np.random.rand() < probabilidade:
                                    nova_grade[ni, nj] = QUEIMANDO
        # Atualizar a grade para o próximo passo
        grade = nova_grade
        grades.append(grade.copy())
    return grades

def calcular_probabilidade_propagacao(params):
    """
    Calcula a probabilidade de propagação do fogo com base nos parâmetros fornecidos.
    """
    # Combinação dos fatores de combustível, climático e terreno
    prob = params['fator_combustivel'] * params['fator_climatico'] * params['fator_terreno']
    # Garantir que a probabilidade esteja entre 0 e 1
    return min(max(prob, 0), 1)

def gerar_relatorio_pdf(resultados):
    """
    Gera um relatório em PDF com os resultados da simulação.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Relatório de Simulação de Propagação de Incêndio", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Data da Simulação: {resultados['data_simulacao']}", ln=True)

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

    # Ajuste dos Parâmetros com os valores obtidos das APIs
    st.header("Ajuste dos Parâmetros")
    st.write("Você pode ajustar os parâmetros manualmente antes de executar a simulação.")

    temperatura = st.slider('Temperatura (°C)', -10.0, 50.0, float(temperatura))
    umidade = st.slider('Umidade Relativa (%)', 0.0, 100.0, float(umidade))
    velocidade_vento = st.slider('Velocidade do Vento (km/h)', 0.0, 100.0, float(velocidade_vento))
    direcao_vento = st.slider('Direção do Vento (graus)', 0.0, 360.0, float(direcao_vento))
    precipitacao = st.slider('Precipitação (mm)', 0.0, 200.0, float(precipitacao))
    radiacao_solar = st.slider('Radiação Solar (W/m²)', 0.0, 1200.0, float(radiacao_solar))
    ndvi = st.slider('NDVI', 0.0, 1.0, float(ndvi))

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
            st.session_state.tipo_solo = tipo_solo
        else:
            st.error("Não foi possível classificar o solo.")
            tipo_solo = 'Desconhecido'
    else:
        tipo_solo = st.session_state.get('tipo_solo', 'Desconhecido')

    # Data da Simulação
    st.header("Data da Simulação")
    data_simulacao = st.date_input("Selecione a data da simulação:", datetime.datetime.now().date())

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

        # Exibição dos parâmetros utilizados
        st.header("Parâmetros Utilizados")
        st.write(pd.DataFrame.from_dict(params, orient='index', columns=['Valor']))

        # Exibição dos resultados
        area_queimada = np.sum(simulacao[-1] == QUEIMADO) * (0.01 ** 2)
        st.header("Resultados da Simulação")
        st.write(f"Data da Simulação: {data_simulacao.strftime('%Y-%m-%d')}")
        st.write(f"Área Queimada (km²): {area_queimada}")

        # Visualização da propagação
        st.header("Mapa de Propagação do Incêndio")
        gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula=0.01)

        # Gerar relatório
        resultados = {
            'data_simulacao': data_simulacao.strftime('%Y-%m-%d'),
            'params': params,
            'resultados': {
                'Área Queimada (km²)': area_queimada
            }
        }
        pdf_bytes = gerar_relatorio_pdf(resultados)
        st.download_button(label="Baixar Relatório em PDF", data=pdf_bytes, file_name="relatorio_simulacao.pdf", mime="application/pdf")
    else:
        st.info("Ajuste os parâmetros e clique em 'Executar Simulação'.")

if __name__ == '__main__':
    main()
