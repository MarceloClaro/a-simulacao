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
import urllib.parse  # For URL encoding

# Initial configuration for Streamlit
st.set_page_config(
    page_title="EcoSim.ai - Fire Propagation Simulator",
    page_icon="🔥",
    layout="wide"
)

# Cell states in the simulation
VIVO = 0        # Unburned vegetation
QUEIMANDO = 1   # Burning vegetation
QUEIMADO = 2    # Burned vegetation

# Colors associated with each state for map visualization
colors = {
    VIVO: 'green',
    QUEIMANDO: 'red',
    QUEIMADO: 'black'
}

# Credentials for API access (ensure to keep these secure)
consumer_key = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
consumer_secret = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'

# Functions for API integration
def obter_token_acesso(consumer_key, consumer_secret):
    """
    Obtains the access token for authentication with Embrapa APIs.
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
        st.error(f"Error obtaining access token: {response.status_code} - {response.text}")
        return None

def obter_ndvi_evi(latitude, longitude, tipo_indice='ndvi', satelite='comb'):
    """
    Obtains the NDVI or EVI time series for the specified location.
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
        # Correct date parsing according to the format returned by the API
        # Display the raw date strings for debugging
        st.write("Raw date strings:", lista_datas)
        try:
            # Adjust the date format based on the actual format returned by the API
            lista_datas = [datetime.strptime(data_str, '%Y-%m-%d') for data_str in lista_datas]
        except ValueError:
            try:
                lista_datas = [datetime.strptime(data_str, '%d/%m/%Y') for data_str in lista_datas]
            except ValueError:
                st.error("Date format not recognized. Please check the date format returned by the API.")
                return None, None
        return lista_ndvi_evi, lista_datas
    else:
        st.error(f"Error obtaining NDVI/EVI data: {response.status_code} - {response.text}")
        return None, None

def obter_dados_climaticos(latitude, longitude):
    """
    Obtains the climatic data time series for the specified location.
    """
    access_token = obter_token_acesso(consumer_key, consumer_secret)
    if access_token is None:
        return None
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
        data_execucao = obter_ultima_data_execucao(var_api, access_token)
        if data_execucao is None:
            continue
        previsao = obter_previsao(var_api, data_execucao, latitude, longitude, access_token)
        if previsao is None:
            continue
        tempos = []
        valores = []
        for ponto in previsao:
            # Convert timestamp in milliseconds to datetime
            tempos.append(datetime.fromtimestamp(ponto['data'] / 1000))
            valor = ponto['valor']
            if param == 'temperatura':
                valor -= 273.15  # Convert from Kelvin to Celsius
            valores.append(valor)
        series_temporais[param] = {'tempos': tempos, 'valores': valores}
    # Process wind speed and direction
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

def obter_ultima_data_execucao(variavel, access_token):
    """
    Obtains the latest execution date available for the specified climatic variable.
    """
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
    """
    Obtains the forecast of the specified climatic variable for the execution date, latitude, and longitude.
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

def classificar_solo(dados_perfil):
    """
    Classifies the soil based on the provided profile data.
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
        st.error(f"Error classifying soil: {response.status_code} - {response.text}")
        return None

# Auxiliary functions
def coordenadas_validas(latitude, longitude):
    """
    Checks if the coordinates are within the limits of South America.
    """
    return -60.0 <= latitude <= 15.0 and -90.0 <= longitude <= -30.0

def gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula):
    """
    Generates the fire propagation map based on the simulation.
    """
    polygons = []
    states = []
    tamanho_grade = simulacao[0].shape[0]
    for i in range(tamanho_grade):
        for j in range(tamanho_grade):
            # Calculate the coordinates of each cell
            x = longitude + (j - tamanho_grade / 2) * tamanho_celula
            y = latitude + (i - tamanho_grade / 2) * tamanho_celula
            # Create a polygon for each cell
            polygon = Polygon([
                (x, y),
                (x + tamanho_celula, y),
                (x + tamanho_celula, y + tamanho_celula),
                (x, y + tamanho_celula)
            ])
            # Get the state of the cell (VIVO, QUEIMANDO, QUEIMADO)
            state = simulacao[-1][i, j]
            polygons.append(polygon)
            states.append(state)

    # Create a GeoDataFrame with the polygons and states
    gdf = gpd.GeoDataFrame({'geometry': polygons, 'state': states}, crs='EPSG:4326')

    # Create the interactive map with Folium
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
    Generates a historical series graph for NDVI/EVI.
    """
    df = pd.DataFrame({'Data': pd.to_datetime(lista_datas), 'Índice': lista_ndvi_evi})
    fig = px.line(df, x='Data', y='Índice', title='Historical Series of NDVI/EVI')
    st.plotly_chart(fig)

def executar_simulacao(params, tamanho_grade, num_passos):
    """
    Executes the fire propagation simulation based on the provided parameters.
    """
    # Initialize the grid with VIVO cells
    grade = np.full((tamanho_grade, tamanho_grade), VIVO)

    # Set the ignition point at the center of the grid
    centro = tamanho_grade // 2
    grade[centro, centro] = QUEIMANDO

    # List to store the grids at each time step
    grades = [grade.copy()]

    # Execute the simulation for 'num_passos' time steps
    for passo in range(num_passos):
        nova_grade = grade.copy()
        for i in range(tamanho_grade):
            for j in range(tamanho_grade):
                if grade[i, j] == QUEIMANDO:
                    # The cell currently burning will be QUEIMADO in the next step
                    nova_grade[i, j] = QUEIMADO
                    # Propagate the fire to neighboring cells
                    vizinhos = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Orthogonal neighbors
                    for vi, vj in vizinhos:
                        ni, nj = i + vi, j + vj
                        # Check if the neighbor is within the grid limits
                        if 0 <= ni < tamanho_grade and 0 <= nj < tamanho_grade:
                            # Only propagate to VIVO cells
                            if grade[ni, nj] == VIVO:
                                # Calculate the propagation probability
                                probabilidade = calcular_probabilidade_propagacao(params)
                                # Decide whether the fire will propagate to the neighboring cell
                                if np.random.rand() < probabilidade:
                                    nova_grade[ni, nj] = QUEIMANDO
        # Update the grid for the next step
        grade = nova_grade
        grades.append(grade.copy())
    return grades

def calcular_probabilidade_propagacao(params):
    """
    Calculates the probability of fire propagation based on the provided parameters.
    """
    # Combine fuel, climatic, and terrain factors
    prob = params['fator_combustivel'] * params['fator_climatico'] * params['fator_terreno']
    # Ensure the probability is between 0 and 1
    return min(max(prob, 0), 1)

def gerar_relatorio_pdf(resultados):
    """
    Generates a PDF report with the simulation results.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Fire Propagation Simulation Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Simulation Date: {resultados['data_simulacao']}", ln=True)

    pdf.cell(200, 10, txt="Parameters Used:", ln=True)
    for key, value in resultados['params'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.cell(200, 10, txt="Results:", ln=True)
    for key, value in resultados['resultados'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

def obter_coordenadas_endereco(endereco):
    """
    Obtains the coordinates (latitude and longitude) for the provided address using the Nominatim API.
    """
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(endereco)}&format=json&limit=1"
    headers = {
        'User-Agent': 'EcoSim.ai/1.0 (contact@example.com)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        resultado = response.json()
        if resultado:
            latitude = float(resultado[0]['lat'])
            longitude = float(resultado[0]['lon'])
            return latitude, longitude
        else:
            st.error("Address not found.")
            return None, None
    else:
        st.error(f"Error querying the geocoding service. Status code: {response.status_code}")
        return None, None

# User interface
def main():
    """
    Main function that controls the user interface and application logic.
    """
    st.title("EcoSim.ai")
    st.subheader("Innovative Fire Propagation Simulator")

    # Section for entering city or address
    st.header("Enter City or Address")
    endereco = st.text_input("Enter the city name or address:")
    if st.button("Get Coordinates"):
        if endereco:
            latitude, longitude = obter_coordenadas_endereco(endereco)
            if latitude and longitude:
                st.success(f"Coordinates obtained: Latitude {latitude:.6f}, Longitude {longitude:.6f}")
                st.session_state.latitude = latitude
                st.session_state.longitude = longitude
            else:
                st.error("Unable to obtain coordinates for the provided address.")
                return
        else:
            st.error("Please enter a valid address.")
            return

    # Or select on the map
    st.header("Or click on the map to select the location")
    m = folium.Map(location=[-15.793889, -47.882778], zoom_start=4)
    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        folium.Marker([st.session_state.latitude, st.session_state.longitude], tooltip="Selected Location").add_to(m)
    map_data = st_folium(m, width=700, height=500)

    if map_data['last_clicked'] is not None:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['lng']
        st.success(f"Selected coordinates: Latitude {latitude:.6f}, Longitude {longitude:.6f}")
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude

    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.warning("Please enter an address or click on the map to select the location.")
        return
    else:
        latitude = st.session_state.latitude
        longitude = st.session_state.longitude

    if not coordenadas_validas(latitude, longitude):
        st.error("The selected coordinates are not within the limits of South America.")
        return

    # Obtain data from APIs
    st.header("Obtaining Data from APIs")
    tipo_indice = st.selectbox('Vegetative Index Type', ['ndvi', 'evi'])
    satelite = st.selectbox('Satellite', ['terra', 'aqua', 'comb'])

    # Obtain NDVI/EVI
    if st.button('Obtain NDVI/EVI from API'):
        lista_ndvi_evi, lista_datas = obter_ndvi_evi(latitude, longitude, tipo_indice, satelite)
        if lista_ndvi_evi is not None:
            ndvi_evi_atual = lista_ndvi_evi[-1]
            st.success(f"Current {tipo_indice.upper()} value: {ndvi_evi_atual}")
            gerar_serie_historica(lista_datas, lista_ndvi_evi)
            st.session_state.ndvi_evi_atual = ndvi_evi_atual
            st.session_state.lista_ndvi_evi = lista_ndvi_evi
            st.session_state.lista_datas_ndvi_evi = lista_datas
            ndvi = ndvi_evi_atual
        else:
            st.error("Unable to obtain NDVI/EVI.")
            return
    else:
        ndvi_evi_atual = st.session_state.get('ndvi_evi_atual', 0.5)
        ndvi = ndvi_evi_atual
        lista_ndvi_evi = st.session_state.get('lista_ndvi_evi', [])
        lista_datas_ndvi_evi = st.session_state.get('lista_datas_ndvi_evi', [])
        if lista_ndvi_evi and lista_datas_ndvi_evi:
            gerar_serie_historica(lista_datas_ndvi_evi, lista_ndvi_evi)

    # Obtain climatic data
    if st.button('Obtain Climatic Data from API'):
        series_temporais = obter_dados_climaticos(latitude, longitude)
        if series_temporais:
            st.success("Climatic data obtained successfully!")
            st.session_state.series_temporais = series_temporais
            # Update manual parameter values with the latest data
            temperatura = series_temporais['temperatura']['valores'][-1]
            umidade = series_temporais['umidade']['valores'][-1]
            velocidade_vento = series_temporais['velocidade_vento']['valores'][-1]
            direcao_vento = series_temporais['direcao_vento']['valores'][-1]
            precipitacao = series_temporais['precipitacao']['valores'][-1]
            radiacao_solar = series_temporais['radiacao_solar']['valores'][-1]
            # Display tables and graphs
            st.header("Climatic Time Series")
            for param, data in series_temporais.items():
                df = pd.DataFrame({'Date': data['tempos'], 'Value': data['valores']})
                st.subheader(f"{param.capitalize()}")
                st.dataframe(df)
                fig = px.line(df, x='Date', y='Value', title=f"Time Series of {param.capitalize()}")
                st.plotly_chart(fig)
        else:
            st.error("Unable to obtain climatic data.")
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

    # Adjust Parameters with values obtained from APIs
    st.header("Adjust Parameters")
    st.write("You can manually adjust the parameters before running the simulation.")

    temperatura = st.slider('Temperature (°C)', -10.0, 50.0, float(temperatura))
    umidade = st.slider('Relative Humidity (%)', 0.0, 100.0, float(umidade))
    velocidade_vento = st.slider('Wind Speed (km/h)', 0.0, 100.0, float(velocidade_vento))
    direcao_vento = st.slider('Wind Direction (degrees)', 0.0, 360.0, float(direcao_vento))
    precipitacao = st.slider('Precipitation (mm)', 0.0, 200.0, float(precipitacao))
    radiacao_solar = st.slider('Solar Radiation (W/m²)', 0.0, 1200.0, float(radiacao_solar))
    ndvi = st.slider('NDVI', 0.0, 1.0, float(ndvi))

    # Soil Classification
    st.header("Soil Classification")
    if st.button('Classify Soil with API'):
        # Soil profile data (simplified for the example)
        dados_perfil = {
            "items": [
                {
                    "ID_PONTO": "Point1",
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
            st.success(f"Soil Classification: {classificacao['items'][0]['ORDEM']}")
            tipo_solo = classificacao['items'][0]['ORDEM']
            st.session_state.tipo_solo = tipo_solo
        else:
            st.error("Unable to classify soil.")
            tipo_solo = 'Unknown'
    else:
        tipo_solo = st.session_state.get('tipo_solo', 'Unknown')

    # Simulation Date
    st.header("Simulation Date")
    data_simulacao = st.date_input("Select the simulation date:", datetime.now().date())

    # Parameters for the simulation
    params = {
        'temperatura': temperatura,
        'umidade': umidade,
        'velocidade_vento': velocidade_vento,
        'direcao_vento': direcao_vento,
        'precipitacao': precipitacao,
        'radiacao_solar': radiacao_solar,
        'ndvi': ndvi,
        'tipo_solo': tipo_solo,
        'fator_combustivel': ndvi,  # Simplified example
        'fator_climatico': (temperatura / 40) * ((100 - umidade) / 100),
        'fator_terreno': 1  # Can include other factors
    }

    # Execute simulation
    if st.button('Run Simulation'):
        tamanho_grade = 50  # Can adjust as needed
        num_passos = 100
        simulacao = executar_simulacao(params, tamanho_grade, num_passos)
        st.success("Simulation completed!")

        # Display used parameters
        st.header("Parameters Used")
        st.write(pd.DataFrame.from_dict(params, orient='index', columns=['Value']))

        # Display simulation results
        area_queimada = np.sum(simulacao[-1] == QUEIMADO) * (0.01 ** 2)
        st.header("Simulation Results")
        st.write(f"Simulation Date: {data_simulacao.strftime('%Y-%m-%d')}")
        st.write(f"Burned Area (km²): {area_queimada}")

        # Visualization of the propagation
        st.header("Fire Propagation Map")
        gerar_mapa_propagacao(simulacao, latitude, longitude, tamanho_celula=0.01)

        # Generate report
        resultados = {
            'data_simulacao': data_simulacao.strftime('%Y-%m-%d'),
            'params': params,
            'resultados': {
                'Burned Area (km²)': area_queimada
            }
        }
        pdf_bytes = gerar_relatorio_pdf(resultados)
        st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="simulation_report.pdf", mime="application/pdf")
    else:
        st.info("Adjust the parameters and click 'Run Simulation'.")

if __name__ == '__main__':
    main()
