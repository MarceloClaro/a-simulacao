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

# Configurações do Streamlit
st.set_page_config(page_title="Simulador de Propagação de Incêndio", page_icon="🔥", layout="wide")

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
def plotar_simulacao(grades):
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

    plt.tight_layout()
    st.pyplot(fig)

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
                'nuvens': hourly_df['Cobertura_Nuvens'].mean()
            }

            st.write("### Configurações da Simulação")
            tamanho_grade = st.slider("Tamanho da Grade", 10, 100, 50)
            passos = st.slider("Número de Passos da Simulação", 10, 200, 100)
            inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
            ruido = st.slider("Nível de Ruído", 1, 100, 10)
            direcao_vento = st.slider("Direção do Vento (graus)", 0, 360, 90)

            if st.button("Executar Simulação de Incêndio"):
                simulacao = executar_simulacao(tamanho_grade, passos, inicio_fogo, params, ruido, direcao_vento)
                plotar_simulacao(simulacao)
                plotar_histogramas_e_margem_erro(simulacao)

if __name__ == '__main__':
    main()
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
from fpdf import FPDF
import base64
import requests
import base64

# Definindo estados das células
VIVO = 0
QUEIMANDO1 = 1
QUEIMANDO2 = 2
QUEIMANDO3 = 3
QUEIMANDO4 = 4
QUEIMADO = 5

# Definindo probabilidades de propagação do fogo para cada estado
probabilidades = {
    VIVO: 0.6,
    QUEIMANDO1: 0.8,
    QUEIMANDO2: 0.8,
    QUEIMANDO3: 0.8,
    QUEIMANDO4: 0.8,
    QUEIMADO: 0
}

# Atribuindo valores numéricos ao tipo de vegetação
valores_tipo_vegetacao = {
    'pastagem': 0.4,
    'matagal': 0.6,
    'floresta decídua': 0.8,
    'floresta tropical': 1.0
}

# Atribuindo valores numéricos ao tipo de solo
valores_tipo_solo = {
    'arenoso': 0.4,
    'misto': 0.6,
    'argiloso': 0.8
}

# Inicializando a matriz do autômato celular
def inicializar_grade(tamanho, inicio_fogo):
    grade = np.zeros((tamanho, tamanho), dtype=int)
    grade[inicio_fogo] = QUEIMANDO1
    return grade

# Calculando a probabilidade de propagação com base nos parâmetros
def calcular_probabilidade_propagacao(params):
    fator_temp = (params['temperatura'] - 20) / 30
    fator_umidade = (100 - params['umidade']) / 100
    fator_velocidade_vento = params['velocidade_vento'] / 50
    fator_densidade_vegetacao = params['densidade_vegetacao'] / 100
    fator_umidade_combustivel = (100 - params['umidade_combustivel']) / 100
    fator_topografia = params['topografia'] / 45
    fator_ndvi = params['ndvi']
    fator_intensidade_fogo = params['intensidade_fogo'] / 10000
    fator_intervencao_humana = 1 - params['intervencao_humana']
    fator_tipo_vegetacao = valores_tipo_vegetacao[params['tipo_vegetacao']]
    fator_tipo_solo = valores_tipo_solo[params['tipo_solo']]

    prob_base = 0.3
    prob = prob_base + 0.1 * (fator_temp + fator_umidade + fator_velocidade_vento + fator_densidade_vegetacao +
                              fator_umidade_combustivel + fator_topografia + fator_ndvi + fator_intensidade_fogo +
                              fator_tipo_vegetacao + fator_tipo_solo) * fator_intervencao_humana

    return min(max(prob, 0), 1)

# Aplicando a regra do autômato celular
def aplicar_regras_fogo(grade, params, ruido):
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
                # Propaga o fogo para células adjacentes com base na probabilidade e efeito do vento
                if grade[i-1, j] == VIVO and np.random.rand() < prob_propagacao * efeito_vento(params['direcao_vento'], (i-1, j), (i, j)) * efeito_ruido(ruido):
                    nova_grade[i-1, j] = QUEIMANDO1
                if grade[i+1, j] == VIVO and np.random.rand() < prob_propagacao * efeito_vento(params['direcao_vento'], (i+1, j), (i, j)) * efeito_ruido(ruido):
                    nova_grade[i+1, j] = QUEIMANDO1
                if grade[i, j-1] == VIVO and np.random.rand() < prob_propagacao * efeito_vento(params['direcao_vento'], (i, j-1), (i, j)) * efeito_ruido(ruido):
                    nova_grade[i, j-1] = QUEIMANDO1
                if grade[i, j+1] == VIVO and np.random.rand() < prob_propagacao * efeito_vento(params['direcao_vento'], (i, j+1), (i, j)) * efeito_ruido(ruido):
                    nova_grade[i, j+1] = QUEIMANDO1
    return nova_grade

# Modelando o efeito do vento
def efeito_vento(direcao_vento, celula, origem):
    angulo_vento_rad = np.deg2rad(direcao_vento)
    vetor_vento = np.array([np.cos(angulo_vento_rad), np.sin(angulo_vento_rad)])
    vetor_direcao = np.array([celula[0] - origem[0], celula[1] - origem[1]])
    vetor_direcao = vetor_direcao / np.linalg.norm(vetor_direcao)
    efeito = np.dot(vetor_vento, vetor_direcao)
    efeito = (efeito + 1) / 2  # Normaliza para um valor entre 0 e 1
    return efeito

# Modelando o efeito do ruído
def efeito_ruido(ruido):
    return 1 + (np.random.rand() - 0.5) * (ruido / 50.0)

# Executando a simulação
def executar_simulacao(tamanho, passos, inicio_fogo, params, ruido):
    grade = inicializar_grade(tamanho, inicio_fogo)
    grades = [grade.copy()]

    for _ in range(passos):
        grade = aplicar_regras_fogo(grade, params, ruido)
        grades.append(grade.copy())

    return grades

# Plotando a simulação
def plotar_simulacao(simulacao, inicio_fogo, direcao_vento):
    num_plots = min(50, len(simulacao))
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.flatten()
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red', 'darkred', 'black'])

    for i, grade in enumerate(simulacao[::max(1, len(simulacao)//num_plots)]):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.imshow(grade, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Passo {i * (len(simulacao)//num_plots)}')

        if i == 0:
            ax.plot(inicio_fogo[1], inicio_fogo[0], 'rs', markersize=5, label='Fogo Inicial')
            ax.legend(loc='upper right')

        if i == len(axes) - 1:
            ax.arrow(90, 90, 10 * np.cos(np.deg2rad(direcao_vento)), 10 * np.sin(np.deg2rad(direcao_vento)),
                     head_width=5, head_length=5, fc='blue', ec='blue')
            ax.text(80, 95, f'Vento {direcao_vento}°', color='blue', fontsize=12)

        ax.grid(True)

    handles = [plt.Rectangle((0,0),1,1, color=cmap.colors[i]) for i in range(6)]
    labels = ['Intacto', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado']
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    st.pyplot(fig)

# Plotando histogramas e gráficos de margem de erro
def plotar_histogramas_e_erros(simulacao):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
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
    st.pyplot(fig)

# Calculando correlações e realizando ANOVA, Q-Exponential e matriz de confusão
def realizar_estatisticas_avancadas(simulacao, params, df_historico_manual):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    valores_params = pd.DataFrame([{
        'temperatura': params['temperatura'],
        'umidade': params['umidade'],
        'velocidade_vento': params['velocidade_vento'],
        'direcao_vento': params['direcao_vento'],
        'precipitacao': params['precipitacao'],
        'radiacao_solar': params['radiacao_solar'],
        'densidade_vegetacao': params['densidade_vegetacao'],
        'umidade_combustivel': params['umidade_combustivel'],
        'topografia': params['topografia'],
        'tipo_solo': valores_tipo_solo[params['tipo_solo']],
        'ndvi': params['ndvi'],
        'intensidade_fogo': params['intensidade_fogo'],
        'tempo_desde_ultimo_fogo': params['tempo_desde_ultimo_fogo'],
        'intervencao_humana': params['intervencao_humana'],
        'ruido': params['ruido']
    }] * len(contagem_queimando_df))
    
    valores_params['Células Queimando'] = contagem_queimando_df['Células Queimando']

    if not df_historico_manual.empty:
        df_historico_manual['tipo_vegetacao'] = df_historico_manual['tipo_vegetacao'].map(valores_tipo_vegetacao)
        df_historico_manual['tipo_solo'] = df_historico_manual['tipo_solo'].map(valores_tipo_solo)
        df_historico_manual = df_historico_manual.apply(pd.to_numeric, errors='coerce')
        valores_params = pd.concat([valores_params, df_historico_manual], ignore_index=True)
        valores_params = valores_params.apply(pd.to_numeric, errors='coerce')

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

# Gerar e baixar PDF
def gerar_pdf(resultados):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resultados da Simulação de Propagação de Incêndio", ln=True, align='C')
    
    for key, value in resultados.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    return pdf.output(dest='S').encode('latin1')

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
        "dataInicial": data_inicial.strftime('%Y-%m-%d'),  # Formato de data
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

# Interface principal do Streamlit
def main():
    st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

    st.title("EcoSim.ai")
    st.subheader("Simulador de Propagação de Incêndio em Autômatos Celulares")

    st.sidebar.image("logo.png", width=200)
    with st.sidebar.expander("Como encontra o NDVI e EVI para Simulação"):
        st.markdown("""Para obter os índices NDVI e EVI da sua região e ajudar na simulação de propagação do fogo, você pode utilizar o **SATVeg - Sistema de Análise Temporal da Vegetação**. Esta ferramenta permite acessar índices vegetativos NDVI e EVI do sensor MODIS em qualquer local da América do Sul.
        Para acessar os dados, visite o site do SATVeg:
        [SATVeg](https://www.satveg.cnptia.embrapa.br/satveg/login.html)""")

    with st.sidebar.expander("Manual de Uso"):
        st.markdown("""### Manual de Uso
        Este simulador permite modelar a propagação do fogo em diferentes condições ambientais. Para utilizar:
        1. Ajuste os parâmetros de simulação usando os controles deslizantes.
        2. Clique em "Executar Simulação" para iniciar a simulação.
        3. Visualize os resultados da propagação do incêndio na área principal.
        """)

    with st.sidebar.expander("Explicação do Processo Matemático"):
        st.markdown("""Olá, sou o Professor Marcelo Claro, especializado em Geografia e Educação Ambiental. Também sou entusiasta em Inteligência Artificial (IA) e Ciências de Dados. Através deste projeto, busco estimular a curiosidade e a iniciação científica entre alunos do ensino básico, promovendo uma abordagem interdisciplinar que desenvolve proficiência digital e inovação. Utilizo diversas técnicas didáticas, como analogias pertinentes, para tornar temas complexos acessíveis e despertar o interesse autodidata nos alunos.""")

    params = {
        'temperatura': st.sidebar.slider('Temperatura (°C)', 0, 50, 30),
        'umidade': st.sidebar.slider('Umidade relativa (%)', 0, 100, 40),
        'velocidade_vento': st.sidebar.slider('Velocidade do Vento (km/h)', 0, 100, 20),
        'direcao_vento': st.sidebar.slider('Direção do Vento (graus)', 0, 360, 90),
        'precipitacao': st.sidebar.slider('Precipitação (mm/dia)', 0, 200, 0),
        'radiacao_solar': st.sidebar.slider('Radiação Solar (W/m²)', 0, 1200, 800),
        'tipo_vegetacao': st.sidebar.selectbox('Tipo de vegetação', ['pastagem', 'matagal', 'floresta decídua', 'floresta tropical']),
        'densidade_vegetacao': st.sidebar.slider('Densidade Vegetal (%)', 0, 100, 70),
        'umidade_combustivel': st.sidebar.slider('Teor de umidade do combustível (%)', 0, 100, 10),
        'topografia': st.sidebar.slider('Topografia (inclinação em graus)', 0, 45, 5),
        'tipo_solo': st.sidebar.selectbox('Tipo de solo', ['arenoso', 'misto', 'argiloso']),
        'ndvi': st.sidebar.slider('NDVI (Índice de Vegetação por Diferença Normalizada)', 0.0, 1.0, 0.6),
        'intensidade_fogo': st.sidebar.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
        'tempo_desde_ultimo_fogo': st.sidebar.slider('Tempo desde o último incêndio (anos)', 0, 100, 10),
        'intervencao_humana': st.sidebar.slider('Fator de Intervenção Humana (escala 0-1)', 0.0, 1.0, 0.2),
        'ruido': st.sidebar.slider('Ruído (%)', 1, 100, 10)
    }

    historico_manual = []
    if st.sidebar.checkbox('Adicionar dados históricos manuais'):
        num_registros = st.sidebar.number_input('Número de registros históricos', min_value=1, max_value=1000, value=3)
        for i in range(num_registros):
            st.write(f"Registro {i+1}")
            registro = {
                'temperatura': st.number_input(f'Temperatura (°C) - {i+1}', 0, 50, 30),
                'umidade': st.number_input(f'Umidade relativa (%) - {i+1}', 0, 100, 40),
                'velocidade_vento': st.number_input(f'Velocidade do Vento (km/h) - {i+1}', 0, 100, 20),
                'direcao_vento': st.number_input(f'Direção do Vento (graus) - {i+1}', 0, 360, 90),
                'precipitacao': st.number_input(f'Precipitação (mm/dia) - {i+1}', 0, 200, 0),
                'radiacao_solar': st.number_input(f'Radiação Solar (W/m²) - {i+1}', 0, 1200, 800),
                'tipo_vegetacao': st.selectbox(f'Tipo de vegetação - {i+1}', ['pastagem', 'matagal', 'floresta decídua', 'floresta tropical']),
                'densidade_vegetacao': st.number_input(f'Densidade Vegetal (%) - {i+1}', 0, 100, 70),
                'umidade_combustivel': st.number_input(f'Teor de umidade do combustível (%) - {i+1}', 0, 100, 10),
                'topografia': st.number_input(f'Topografia (inclinação em graus) - {i+1}', 0, 45, 5),
                'tipo_solo': st.selectbox(f'Tipo de solo - {i+1}', ['arenoso', 'misto', 'argiloso']),
                'ndvi': st.number_input(f'NDVI (Índice de Vegetação por Diferença Normalizada) - {i+1}', 0.0, 1.0, 0.6),
                'intensidade_fogo': st.number_input(f'Intensidade do Fogo (kW/m) - {i+1}', 0, 10000, 5000),
                'tempo_desde_ultimo_fogo': st.number_input(f'Tempo desde o último incêndio (anos) - {i+1}', 0, 100, 10),
                'intervencao_humana': st.number_input(f'Fator de Intervenção Humana (escala 0-1) - {i+1}', 0.0, 1.0, 0.2),
                'ruido': st.number_input(f'Ruído (%) - {i+1}', 1, 100, 10)
            }
            historico_manual.append(registro)

    df_historico_manual = pd.DataFrame(historico_manual)

    tamanho_grade = st.sidebar.slider('Tamanho da grade', 10, 100, 50)
    num_passos = st.sidebar.slider('Número de passos', 10, 200, 100)

    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""Projeto Geomaker + IA 
    - Professor: Marcelo Claro.
    Contatos: marceloclaro@gmail.com
    Whatsapp: (88)981587145
    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)""")

    # Controle de Áudio
    st.sidebar.title("Controle de Áudio")
    mp3_files = {"Explicação do Processo Matemático": "apresentação ac.mp3"}
    selected_mp3 = st.sidebar.radio("Escolha uma música", list(mp3_files.keys()))
    loop = st.sidebar.checkbox("Repetir música")

    audio_placeholder = st.sidebar.empty()
    if selected_mp3:
        mp3_path = mp3_files[selected_mp3]
        try:
            with open(mp3_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                loop_attr = "loop" if loop else ""
                audio_html = f"""
                <audio id="audio-player" controls autoplay {loop_attr}>
                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                  Seu navegador não suporta o elemento de áudio.
                </audio>
                """
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        except FileNotFoundError:
            audio_placeholder.error(f"Arquivo {mp3_path} não encontrado.")

    if st.button('Executar Simulação'):
        inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
        ruido = params['ruido']
        simulacao = executar_simulacao(tamanho_grade, num_passos, inicio_fogo, params, ruido)
        plotar_simulacao(simulacao, inicio_fogo, params['direcao_vento'])
        plotar_histogramas_e_erros(simulacao)
        correlacao_spearman, f_val, p_val, valores_q_exponencial, valores_q_estatistica, matriz_confusao = realizar_estatisticas_avancadas(simulacao, params, df_historico_manual)

        resultados = {
            "Matriz de Correlação (Spearman)": correlacao_spearman.to_string(),
            "F-valor ANOVA": f_val,
            "p-valor ANOVA": p_val,
            "Valores Q-Exponencial": valores_q_exponencial.to_string(),
            "Valores Q-Estatística": valores_q_estatistica,
            "Matriz de Confusão": matriz_confusao.tolist()
        }

        pdf_bytes = gerar_pdf(resultados)
        st.download_button(label="Baixar PDF", data=pdf_bytes, file_name="resultados_simulacao.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
