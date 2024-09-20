import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Para gráficos interativos
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
from fpdf import FPDF
import base64
import requests
import folium  # Para o mapa interativo
from streamlit_folium import st_folium

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

# Função para obter o token de acesso da API SATVeg MODIS
def obter_token_acesso():
    consumer_key = '8DEyf0gKWuBsN75KRcjQIc4c03Ea'
    consumer_secret = 'bxY5z5ZnwKefqPmka3MLKNb0vJMa'
    token_url = 'https://api.cnptia.embrapa.br/token'

    # Codifica o consumer key e secret em Base64
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

# Função para obter NDVI ou EVI da API com base na latitude e longitude
def obter_ndvi_evi(latitude, longitude, tipo_indice='ndvi', satelite='comb'):
    access_token = obter_token_acesso()
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
        ax.axis('off')

    handles = [plt.Rectangle((0,0),1,1, color=cmap.colors[i]) for i in range(6)]
    labels = ['Intacto', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado']
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    st.pyplot(fig)

# Plotando histogramas e gráficos de margem de erro
def plotar_histogramas_e_erros(simulacao):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    # Gráfico interativo usando Plotly
    fig = px.line(contagem_queimando_df, y="Células Queimando", title='Células Queimando ao Longo do Tempo')
    st.plotly_chart(fig)

    # Histograma interativo
    fig_hist = px.histogram(contagem_queimando_df, x="Células Queimando", nbins=20, title='Histograma de Células Queimando')
    st.plotly_chart(fig_hist)

# Calculando correlações e realizando análises estatísticas avançadas
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
    st.dataframe(correlacao_spearman.style.background_gradient(cmap='coolwarm').set_precision(2))

    # ANOVA
    tercios = np.array_split(contagem_queimando_df["Células Queimando"], 3)
    f_val, p_val = stats.f_oneway(tercios[0], tercios[1], tercios[2])
    st.write("### Resultado da ANOVA:")
    st.write(f"F-valor: {f_val:.2f}, p-valor: {p_val:.4f}")

    # Matriz de Confusão
    y_true = np.concatenate([grade.flatten() for grade in simulacao[:-1]])
    y_pred = np.concatenate([grade.flatten() for grade in simulacao[1:]])
    matriz_confusao = confusion_matrix(y_true, y_pred, labels=[VIVO, QUEIMANDO1, QUEIMANDO2, QUEIMANDO3, QUEIMANDO4, QUEIMADO])

    st.write("### Matriz de Confusão:")
    df_matriz_confusao = pd.DataFrame(matriz_confusao, index=['Vivo', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado'],
                                      columns=['Vivo', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado'])
    st.dataframe(df_matriz_confusao.style.background_gradient(cmap='Blues'))

    return correlacao_spearman, f_val, p_val, matriz_confusao

# Gerar e baixar PDF
def gerar_pdf(resultados):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resultados da Simulação de Propagação de Incêndio", ln=True, align='C')

    for key, value in resultados.items():
        if isinstance(value, pd.DataFrame):
            pdf.cell(200, 10, txt=f"{key}:", ln=True)
            # Converter DataFrame em texto
            pdf.multi_cell(0, 10, value.to_string())
        else:
            pdf.multi_cell(0, 10, f"{key}: {value}")

    return pdf.output(dest='S').encode('latin1')

# Função para validar coordenadas
def coordenadas_validas(latitude, longitude):
    return -60.0 <= latitude <= 15.0 and -90.0 <= longitude <= -30.0

# Interface principal do Streamlit
def main():
    st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

    st.title("EcoSim.ai")
    st.subheader("Simulador de Propagação de Incêndio em Autômatos Celulares")

    # Seção de parâmetros da simulação
    st.sidebar.title("Parâmetros da Simulação")

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
        'ndvi': st.sidebar.slider('NDVI (Índice de Vegetação)', 0.0, 1.0, 0.6),
        'intensidade_fogo': st.sidebar.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
        'tempo_desde_ultimo_fogo': st.sidebar.slider('Tempo desde o último incêndio (anos)', 0, 100, 10),
        'intervencao_humana': st.sidebar.slider('Fator de Intervenção Humana (0-1)', 0.0, 1.0, 0.2),
        'ruido': st.sidebar.slider('Ruído (%)', 1, 100, 10)
    }

    tamanho_grade = st.sidebar.slider('Tamanho da grade', 10, 100, 50)
    num_passos = st.sidebar.slider('Número de passos', 10, 200, 100)

    # Seção para seleção de coordenadas no mapa
    st.sidebar.title("Seleção de Localização")
    st.sidebar.write("Clique no mapa para selecionar a localização desejada.")
    m = folium.Map(location=[-15.793889, -47.882778], zoom_start=4)
    map_data = st_folium(m, width=350, height=250)
    
    # Verifica se o usuário clicou no mapa
    if map_data['last_clicked'] is not None:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['lng']
        st.sidebar.write(f"Latitude: {latitude:.6f}")
        st.sidebar.write(f"Longitude: {longitude:.6f}")
    else:
        latitude = None
        longitude = None

    tipo_indice = st.sidebar.selectbox('Tipo de Índice Vegetativo', ['ndvi', 'evi'])
    satelite = st.sidebar.selectbox('Satélite', ['terra', 'aqua', 'comb'])

    # Botão para obter NDVI/EVI da API com validação
    if st.sidebar.button('Obter NDVI/EVI da API'):
        if latitude is not None and longitude is not None:
            if coordenadas_validas(latitude, longitude):
                lista_ndvi_evi, lista_datas = obter_ndvi_evi(latitude, longitude, tipo_indice, satelite)
                if lista_ndvi_evi is not None:
                    ndvi_evi_atual = lista_ndvi_evi[-1]  # Obtém o valor mais recente
                    st.sidebar.success(f"Valor atual de {tipo_indice.upper()}: {ndvi_evi_atual}")
                    params['ndvi'] = ndvi_evi_atual  # Atualiza o parâmetro 'ndvi' nos parâmetros da simulação
            else:
                st.sidebar.error("As coordenadas inseridas não estão dentro dos limites da América do Sul. Por favor, selecione um ponto válido no mapa.")
        else:
            st.sidebar.error("Por favor, clique no mapa para selecionar a localização.")

    # Botão para executar a simulação
    if st.sidebar.button('Executar Simulação'):
        with st.spinner('Executando a simulação, por favor aguarde...'):
            inicio_fogo = (tamanho_grade // 2, tamanho_grade // 2)
            ruido = params['ruido']
            simulacao = executar_simulacao(tamanho_grade, num_passos, inicio_fogo, params, ruido)
            st.success("Simulação concluída!")
            plotar_simulacao(simulacao, inicio_fogo, params['direcao_vento'])
            plotar_histogramas_e_erros(simulacao)
            correlacao_spearman, f_val, p_val, matriz_confusao = realizar_estatisticas_avancadas(simulacao, params, pd.DataFrame())

            resultados = {
                "Matriz de Correlação (Spearman)": correlacao_spearman,
                "F-valor ANOVA": f"{f_val:.2f}",
                "p-valor ANOVA": f"{p_val:.4f}",
                "Matriz de Confusão": pd.DataFrame(matriz_confusao)
            }

            pdf_bytes = gerar_pdf(resultados)
            st.download_button(label="Baixar Resultados em PDF", data=pdf_bytes, file_name="resultados_simulacao.pdf", mime="application/pdf")
    else:
        st.info("Ajuste os parâmetros e clique em 'Executar Simulação'.")

    # Seção de Créditos
    st.sidebar.write("---")
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [@marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    """)

if __name__ == "__main__":
    main()
