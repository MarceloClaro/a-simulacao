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
from fpdf import FPDF
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Chaves de API da Embrapa
EMBRAPA_CONSUMER_KEY = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
EMBRAPA_CONSUMER_SECRET = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

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
        series = pd.DataFrame({
            'Data': pd.to_datetime(data['listaDatas']),
            tipo_indice.upper(): data['listaSerie']
        })
        return series
    else:
        st.error(f"Erro ao obter NDVI/EVI: {response.status_code} - Detalhes: {response.json().get('user_message', '')}")
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
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_speed_100m",
            "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm", "cloud_cover", "precipitation"
        ]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Processamento dos dados horários
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
        "Velocidade_Vento_100m": hourly.Variables(3).ValuesAsNumpy(),
        "Temperatura_Solo_0_7cm": hourly.Variables(4).ValuesAsNumpy(),
        "Temperatura_Solo_7_28cm": hourly.Variables(5).ValuesAsNumpy(),
        "Umidade_Solo_0_7cm": hourly.Variables(6).ValuesAsNumpy(),
        "Umidade_Solo_7_28cm": hourly.Variables(7).ValuesAsNumpy(),
        "Cobertura_Nuvens": hourly.Variables(8).ValuesAsNumpy(),
        "Precipitacao": hourly.Variables(9).ValuesAsNumpy()
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
        st.error("Endereço não encontrado ou fora da América do Sul.")
        return None, None

# Função para simulação de propagação de incêndio usando autômatos celulares
VIVO, QUEIMANDO1, QUEIMANDO2, QUEIMANDO3, QUEIMANDO4, QUEIMADO = 0, 1, 2, 3, 4, 5

def inicializar_grade(tamanho, inicio_fogo):
    grade = np.zeros((tamanho, tamanho), dtype=int)
    grade[inicio_fogo] = QUEIMANDO1
    return grade

# Função para calcular a probabilidade de propagação de acordo com os parâmetros
def calcular_probabilidade_propagacao(params, direcao_vento):
    fatores = {
        "temp": (params['temperatura'] - 20) / 30,
        "umidade": (100 - params['umidade']) / 100,
        "vento_10m": params['vento_10m'] / 50,
        "vento_100m": params['vento_100m'] / 50,
        "ndvi": params['ndvi'],
        "evi": params['evi'],
        "chuva": (50 - params['chuva']) / 50,
        "nuvens": (100 - params['nuvens']) / 100,
        "direcao_vento": (direcao_vento / 360)  # Normalizando a direção do vento
    }
    prob_base = 0.3
    prob = prob_base + 0.1 * (fatores["temp"] + fatores["umidade"] + fatores["vento_10m"] + fatores["vento_100m"] + fatores["ndvi"] + fatores["evi"] + fatores["chuva"] + fatores["nuvens"])
    return min(max(prob, 0), 1)

# Função para aplicar regras de propagação do fogo
def aplicar_regras_fogo(grade, params, ruido, direcao_vento):
    nova_grade = grade.copy()
    tamanho = grade.shape[0]
    prob_propagacao = calcular_probabilidade_propagacao(params, direcao_vento)

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
                vizinhos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in vizinhos:
                    # Implementando a influência da direção do vento
                    if grade[ni, nj] == VIVO:
                        prob_ajustada = prob_propagacao * (1 + np.cos(np.radians(direcao_vento - 90)) / 2)  # Ajusta a probabilidade conforme a direção do vento
                        if np.random.rand() < prob_ajustada * (1 + ruido / 50.0):
                            nova_grade[ni, nj] = QUEIMANDO1
    return nova_grade

# Função para executar a simulação
def executar_simulacao(tamanho, passos, inicio_fogo, params, ruido, direcao_vento):
    grade = inicializar_grade(tamanho, inicio_fogo)
    grades = [grade.copy()]
    for _ in range(passos):
        grade = aplicar_regras_fogo(grade, params, ruido, direcao_vento)
        grades.append(grade.copy())
    return grades

# Função para exibir gráficos de histogramas e margem de erro
def plotar_histogramas_e_margem_erro(simulacao):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]

    st.sidebar.write("### Histograma de Células Queimando")
    fig, ax = plt.subplots()
    ax.hist(contagem_queimando, bins=20, color='orange')
    ax.set_title("Histograma de Células Queimando")
    ax.set_xlabel("Número de Células Queimando")
    ax.set_ylabel("Frequência")
    st.sidebar.pyplot(fig)

    # Gráfico de média e margem de erro
    st.sidebar.write("### Média e Margem de Erro")
    media_movel = pd.Series(contagem_queimando).rolling(window=5).mean()
    std_movel = pd.Series(contagem_queimando).rolling(window=5).std()

    fig, ax = plt.subplots()
    ax.plot(media_movel, label='Média', color='blue')
    ax.fill_between(media_movel.index, media_movel - std_movel, media_movel + std_movel, color='blue', alpha=0.2)
    ax.set_title("Média e Margem de Erro")
    ax.set_xlabel("Passos da Simulação")
    ax.set_ylabel("Número de Células Queimando")
    ax.legend()
    st.sidebar.pyplot(fig)

# Função para plotar a simulação de incêndio
def plotar_simulacao(grades, inicio_fogo, direcao_vento):
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.flatten()
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red', 'darkred', 'black'])

    for i, grade in enumerate(grades[::max(1, len(grades)//50)]):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.imshow(grade, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Passo {i}')
        ax.grid(False)

        # Marcar o ponto de fogo inicial
        if i == 0:
            ax.plot(inicio_fogo[1], inicio_fogo[0], 'rs', markersize=5, label='Fogo Inicial')

        # Adicionando seta para direção do vento
        if i == len(axes) - 1:
            ax.arrow(90, 90, 10 * np.cos(np.deg2rad(direcao_vento)), 10 * np.sin(np.deg2rad(direcao_vento)),
                     head_width=2, head_length=3, fc='blue', ec='blue', label='Direção do Vento')
            ax.text(80, 95, f'Vento {direcao_vento}°', color='blue', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

# Função para calcular correlações e realizar ANOVA, Q-Estatística e matriz de confusão
def realizar_estatisticas_avancadas(simulacao, params, df_historico_manual):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    valores_params = pd.DataFrame([{
        'temperatura': params['temperatura'],
        'umidade': params['umidade'],
        'vento_10m': params['vento_10m'],
        'vento_100m': params['vento_100m'],
        'ndvi': params['ndvi'],
        'evi': params['evi'],
        'chuva': params['chuva'],
        'nuvens': params['nuvens'],
        'ruido': params['ruido']
    }] * len(contagem_queimando_df))
    
    valores_params['Células Queimando'] = contagem_queimando_df['Células Queimando']

    if not df_historico_manual.empty:
        valores_params = pd.concat([valores_params, df_historico_manual], ignore_index=True)

    correlacao_spearman = valores_params.corr(method='spearman')
    st.write("### Matriz de Correlação (Spearman):")
    st.write(correlacao_spearman)

    tercios = np.array_split(contagem_queimando_df["Células Queimando"], 3)
    f_val, p_val = stats.f_oneway(tercios[0], tercios[1], tercios[2])
    st.write("### Resultado da ANOVA:")
    st.write(f"F-valor: {f_val}, p-valor: {p_val}")

    def q_exponencial(valores, q):
        return (1 - (1 - q) * valores)**(1 / (1 - q))

    q_valor = 1.5
    valores_q_exponencial = q_exponencial(contagem_queimando_df["Células Queimando"], q_valor)
    st.write("### Valores Q-Exponencial:")
    st.write(valores_q_exponencial)

    def q_estatistica(valores, q):
        return np.sum((valores_q_exponencial - np.mean(valores_q_exponencial))**2) / len(valores_q_exponencial)

    valores_q_estatistica = q_estatistica(contagem_queimando_df["Células Queimando"], q_valor)
    st.write("### Valores Q-Estatística:")
    st.write(valores_q_estatistica)

    y_true = np.concatenate([grade.flatten() for grade in simulacao[:-1]])
    y_pred = np.concatenate([grade.flatten() for grade in simulacao[1:]])
    matriz_confusao = confusion_matrix(y_true, y_pred, labels=[VIVO, QUEIMANDO1, QUEIMANDO2, QUEIMANDO3, QUEIMANDO4, QUEIMADO])
    st.write("### Matriz de Confusão:")
    st.write(matriz_confusao)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['Vivo', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado'], yticklabels=['Vivo', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado'])
    ax.set_xlabel('Estado Previsto')
    ax.set_ylabel('Estado Real')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

    return correlacao_spearman, f_val, p_val, valores_q_exponencial, valores_q_estatistica, matriz_confusao

# Interface do usuário
def main():
    st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

    st.title("EcoSim.ai")
    st.subheader("Simulador de Propagação de Incêndio em Autômatos Celulares")

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
            st.dataframe(hourly_df)
            st.write("### NDVI (Embrapa)")
            st.dataframe(ndvi_df)
            st.write("### EVI (Embrapa)")
            st.dataframe(evi_df)

            params = {
                'temperatura': hourly_df['Temperatura_2m'].mean(),
                'umidade': hourly_df['Umidade_Relativa_2m'].mean(),
                'vento_10m': hourly_df['Velocidade_Vento_10m'].mean(),
                'vento_100m': hourly_df['Velocidade_Vento_100m'].mean(),
                'ndvi': ndvi_df['NDVI'].mean(),
                'evi': evi_df['EVI'].mean(),
                'chuva': hourly_df['Precipitacao'].sum(),
                'nuvens': hourly_df['Cobertura_Nuvens'].mean(),
                'ruido': st.slider("Nível de Ruído", 1, 100, 10)
            }

            st.write("### Configurações da Simulação")
            tamanho_grade = st.slider("Tamanho da Grade", 10, 100, 50)
            passos = st.slider("Número de Passos da Simulação", 10, 200, 100)
            inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
            direcao_vento = st.slider("Direção do Vento (graus)", 0, 360, 90)

            if st.button("Executar Simulação de Incêndio"):
                simulacao = executar_simulacao(tamanho_grade, passos, inicio_fogo, params, params['ruido'], direcao_vento)
                plotar_simulacao(simulacao, inicio_fogo, direcao_vento)
                plotar_histogramas_e_margem_erro(simulacao)
                correlacao_spearman, f_val, p_val, valores_q_exponencial, valores_q_estatistica, matriz_confusao = realizar_estatisticas_avancadas(simulacao, params, pd.DataFrame())

if __name__ == "__main__":
    main()
