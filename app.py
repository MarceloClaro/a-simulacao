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

# Interface principal do Streamlit
def main():
    st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

    st.title("EcoSim.ai")
    st.subheader("Simulador de Propagação de Incêndio em Autômatos Celulares")

    st.sidebar.image("logo.png", width=200)

    with st.sidebar.expander("Manual de Uso"):
        st.markdown("""
        ### Manual de Uso
        Este simulador permite modelar a propagação do fogo em diferentes condições ambientais. Para utilizar:
        1. Ajuste os parâmetros de simulação usando os controles deslizantes.
        2. Clique em "Executar Simulação" para iniciar a simulação.
        3. Visualize os resultados da propagação do incêndio na área principal.
        """)

    with st.sidebar.expander("Explicação do Processo Matemático"):
        st.markdown("""
        ### Explicação do Processo Matemático
        O simulador utiliza autômatos celulares para modelar a propagação do fogo. Cada célula do grid representa um pedaço de terreno que pode estar em diferentes estados:
        - **Intacto**: Vegetação não queimada.
        - **Queimando1 a Queimando4**: Diferentes estágios de queima.
        - **Queimado**: Vegetação queimada.
        """)

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
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)

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
