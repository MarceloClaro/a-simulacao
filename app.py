import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
from fpdf import FPDF

# Define estados das células
VIVO = 0        # Célula viva (verde)
QUEIMANDO1 = 1  # Célula começando a queimar (amarelo)
QUEIMANDO2 = 2  # Célula continuando a queimar (laranja)
QUEIMANDO3 = 3  # Célula continuando a queimar (vermelho)
QUEIMANDO4 = 4  # Célula continuando a queimar (vermelho escuro)
QUEIMADO = 5    # Célula queimada (preto)

# Define as probabilidades de propagação do fogo para cada estado
probabilidades = {
    VIVO: 0.6,       # Probabilidade de uma célula viva pegar fogo
    QUEIMANDO1: 0.8, # Probabilidade de uma célula queimando continuar queimando
    QUEIMANDO2: 0.8, # Continuação da queima
    QUEIMANDO3: 0.8, # Continuação da queima
    QUEIMANDO4: 0.8, # Continuação da queima
    QUEIMADO: 0      # Uma célula queimada não pode pegar fogo novamente
}

# Atribui valores numéricos ao tipo de vegetação
valores_tipo_vegetacao = {
    'pastagem': 0.4,
    'matagal': 0.6,
    'floresta decídua': 0.8,
    'floresta tropical': 1.0
}

# Atribui valores numéricos ao tipo de solo
valores_tipo_solo = {
    'arenoso': 0.4,
    'misto': 0.6,
    'argiloso': 0.8
}

# Inicializa a matriz do autômato celular
def inicializar_grade(tamanho, inicio_fogo):
    grade = np.zeros((tamanho, tamanho), dtype=int)  # Cria uma matriz de zeros (células vivas)
    grade[inicio_fogo] = QUEIMANDO1  # Define a célula inicial como queimando
    return grade

# Função para calcular a probabilidade de propagação com base nos parâmetros
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

# Aplica a regra do autômato celular
def aplicar_regras_fogo(grade, params, ruido):
    nova_grade = grade.copy()  # Cria uma cópia da matriz para atualizar os estados
    tamanho = grade.shape[0]  # Obtém o tamanho da matriz
    prob_propagacao = calcular_probabilidade_propagacao(params)

    for i in range(1, tamanho - 1):  # Percorre cada célula (ignorando bordas)
        for j in range(1, tamanho - 1):
            if grade[i, j] == QUEIMANDO1:
                nova_grade[i, j] = QUEIMANDO2  # Atualiza célula para o próximo estado de queima
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

# Função para modelar o efeito do vento
def efeito_vento(direcao_vento, celula, origem):
    angulo_vento_rad = np.deg2rad(direcao_vento)
    vetor_vento = np.array([np.cos(angulo_vento_rad), np.sin(angulo_vento_rad)])
    
    vetor_direcao = np.array([celula[0] - origem[0], celula[1] - origem[1]])
    vetor_direcao = vetor_direcao / np.linalg.norm(vetor_direcao)
    
    efeito = np.dot(vetor_vento, vetor_direcao)
    efeito = (efeito + 1) / 2  # Normaliza para um valor entre 0 e 1
    
    return efeito

# Função para modelar o efeito do ruído
def efeito_ruido(ruido):
    return 1 + (np.random.rand() - 0.5) * (ruido / 50.0)

# Função para executar a simulação
def executar_simulacao(tamanho, passos, inicio_fogo, params, ruido):
    grade = inicializar_grade(tamanho, inicio_fogo)  # Inicializa a matriz do autômato celular
    grades = [grade.copy()]  # Cria uma lista para armazenar os estados em cada passo

    for _ in range(passos):  # Executa a simulação para o número de passos definido
        grade = aplicar_regras_fogo(grade, params, ruido)  # Aplica as regras do autômato
        grades.append(grade.copy())  # Armazena a matriz atualizada na lista

    return grades

# Função para plotar a simulação
def plotar_simulacao(simulacao, inicio_fogo, direcao_vento):
    num_plots = min(50, len(simulacao))  # Define o número máximo de gráficos a serem plotados
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))  # Cria um grid de subplots (5 linhas, 10 colunas)
    axes = axes.flatten()  # Achata a matriz de eixos para fácil iteração

    # Define um mapa de cores personalizado para os diferentes estados das células
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red', 'darkred', 'black'])

    # Itera sobre os estados da simulação para plotar cada um
    for i, grade in enumerate(simulacao[::max(1, len(simulacao)//num_plots)]):
        if i >= len(axes):  # Verifica se o número máximo de gráficos foi atingido
            break
        ax = axes[i]
        ax.imshow(grade, cmap=cmap, interpolation='nearest')  # Plota a matriz atual com o mapa de cores
        ax.set_title(f'Passo {i * (len(simulacao)//num_plots)}')  # Define o título do subplot com o passo da simulação

        # Marca o quadrinho inicial com um quadrado vermelho
        if i == 0:
            ax.plot(inicio_fogo[1], inicio_fogo[0], 'rs', markersize=5, label='Fogo Inicial')
            ax.legend(loc='upper right')

        # Desenha uma seta para indicar a direção do vento com texto
        if i == len(axes) - 1:  # Último gráfico
            ax.arrow(90, 90, 10 * np.cos(np.deg2rad(direcao_vento)), 10 * np.sin(np.deg2rad(direcao_vento)),
                     head_width=5, head_length=5, fc='blue', ec='blue')
            ax.text(80, 95, f'Vento {direcao_vento}°', color='blue', fontsize=12)

        ax.grid(True)  # Exibe a malha cartesiana

    # Cria a legenda para os diferentes estados das células
    handles = [plt.Rectangle((0,0),1,1, color=cmap.colors[i]) for i in range(6)]
    labels = ['Intacto', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado']
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    st.pyplot(fig)

# Função para plotar histogramas e gráficos de margem de erro
def plotar_histogramas_e_erros(simulacao):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma
    sns.histplot(contagem_queimando_df, x="Células Queimando", ax=ax[0], kde=True, bins=20, color='orange')
    ax[0].set_title('Histograma de Células Queimando')
    ax[0].set_xlabel('Número de Células Queimando')
    ax[0].set_ylabel('Frequência')
    
    # Gráfico de média e margem de erro
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

# Função para calcular correlações e realizar ANOVA, Q-Exponential e matriz de confusão
def realizar_estatisticas_avancadas(simulacao, params, df_historico_manual):
    contagem_queimando = [np.sum(grade == QUEIMANDO1) + np.sum(grade == QUEIMANDO2) + np.sum(grade == QUEIMANDO3) + np.sum(grade == QUEIMANDO4) for grade in simulacao]
    contagem_queimando_df = pd.DataFrame(contagem_queimando, columns=["Células Queimando"])

    # Adiciona os parâmetros à análise
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

    # Adiciona dados históricos manuais
    if not df_historico_manual.empty:
        df_historico_manual['tipo_vegetacao'] = df_historico_manual['tipo_vegetacao'].map(valores_tipo_vegetacao)
        df_historico_manual['tipo_solo'] = df_historico_manual['tipo_solo'].map(valores_tipo_solo)
        df_historico_manual = df_historico_manual.apply(pd.to_numeric, errors='coerce')
        valores_params = pd.concat([valores_params, df_historico_manual], ignore_index=True)
        valores_params = valores_params.apply(pd.to_numeric, errors='coerce')

    # Correlação de Spearman
    correlacao_spearman = valores_params.corr(method='spearman')
    st.write("### Matriz de Correlação (Spearman):")
    st.write(correlacao_spearman)

    # ANOVA
    tercios = np.array_split(contagem_queimando_df["Células Queimando"], 3)
    f_val, p_val = stats.f_oneway(tercios[0], tercios[1], tercios[2])
    st.write("### Resultado da ANOVA:")
    st.write(f"F-valor: {f_val}, p-valor: {p_val}")

    # Q-Exponential
    def q_exponencial(valores, q):
        return (1 - (1 - q) * valores)**(1 / (1 - q))

    q_valor = 1.5  # Exemplo de valor de q
    valores_q_exponencial = q_exponencial(contagem_queimando_df["Células Queimando"], q_valor)
    st.write("### Valores Q-Exponencial:")
    st.write(valores_q_exponencial)

    # Estatística Q
    def q_estatistica(valores, q):
        return np.sum((valores_q_exponencial - np.mean(valores_q_exponencial))**2) / len(valores_q_exponencial)

    valores_q_estatistica = q_estatistica(contagem_queimando_df["Células Queimando"], q_valor)
    st.write("### Valores Q-Estatística:")
    st.write(valores_q_estatistica)

    # Matriz de Confusão
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

# Função para gerar e baixar PDF
def gerar_pdf(resultados):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resultados da Simulação de Propagação de Incêndio", ln=True, align='C')
    
    for key, value in resultados.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    return pdf.output(dest='S').encode('latin1')

# Função principal para a interface do Streamlit
def main():
    st.set_page_config(page_title="EcoSim.ai - Simulador de Propagação de Incêndio", page_icon="🔥")

    st.title("EcoSim.ai")
    st.subheader("Simulador de Propagação de Incêndio em Autômatos Celulares")

    st.sidebar.image("logo.png", width=200)

    # Manual de uso
    with st.sidebar.expander("Manual de Uso"):
        st.markdown("""
        ### Manual de Uso
        Este simulador permite modelar a propagação do fogo em diferentes condições ambientais. Para utilizar:
        1. Ajuste os parâmetros de simulação usando os controles deslizantes.
        2. Clique em "Executar Simulação" para iniciar a simulação.
        3. Visualize os resultados da propagação do incêndio na área principal.

        ### Parâmetros de Simulação
        - **Temperatura (°C)**: Define a temperatura ambiente.
        - **Umidade relativa (%)**: Define a umidade do ar.
        - **Velocidade do Vento (km/h)**: Define a velocidade do vento.
        - **Direção do Vento (graus)**: Define a direção do vento em graus.
        - **Precipitação (mm/dia)**: Define a quantidade de chuva.
        - **Radiação Solar (W/m²)**: Define a intensidade da radiação solar.
        - **Tipo de vegetação**: Seleciona o tipo de vegetação.
        - **Densidade Vegetal (%)**: Define a densidade da vegetação.
        - **Teor de umidade do combustível (%)**: Define a umidade do material combustível.
        - **Topografia (inclinação em graus)**: Define a inclinação do terreno.
        - **Tipo de solo**: Seleciona o tipo de solo.
        - **NDVI**: Define o índice de vegetação por diferença normalizada.
        - **Intensidade do Fogo (kW/m)**: Define a intensidade do fogo.
        - **Tempo desde o último incêndio (anos)**: Define o tempo desde o último incêndio.
        - **Fator de Intervenção Humana**: Define a intervenção humana na propagação do fogo.
        - **Ruído (%)**: Define a aleatoriedade no modelo de propagação do fogo.
        """)

    # Explicação do processo matemático e estatísticas
    with st.sidebar.expander("Explicação do Processo Matemático"):
        st.markdown("""
        ### Explicação do Processo Matemático
        O simulador utiliza autômatos celulares para modelar a propagação do fogo. Cada célula do grid representa um pedaço de terreno que pode estar em diferentes estados:
        - **Intacto**: Vegetação não queimada.
        - **Queimando1 a Queimando4**: Diferentes estágios de queima.
        - **Queimado**: Vegetação queimada.

        A probabilidade de uma célula pegar fogo depende de vários fatores, como temperatura, umidade, velocidade e direção do vento, e densidade da vegetação. O efeito do vento é modelado usando vetores direcionais e a propagação do fogo é calculada a cada passo de tempo da simulação.

        O parâmetro de **ruído** adiciona uma aleatoriedade à propagação do fogo, representando incertezas e variabilidades no ambiente que podem afetar o comportamento do incêndio.

        ### Equação da Regra do Autômato Celular
        A probabilidade de uma célula (i, j) pegar fogo é dada por:

        \\[
        P_{spread}(i, j) = P_{base} \times W_{effect}(i, j) \times N_{effect}
        \\]

        Onde:
        - \\( P_{spread}(i, j) \\) é a probabilidade de propagação do fogo para a célula (i, j).
        - \\( P_{base} \\) é a probabilidade base de uma célula pegar fogo (dependente do estado da célula e de outros fatores).
        - \\( W_{effect}(i, j) \\) é o efeito do vento na propagação do fogo para a célula (i, j).
        - \\( N_{effect} \\) é o efeito do ruído na propagação do fogo.

        ### Elementos da Equação
        - **P_{base}**: Esta é a probabilidade base determinada por vários fatores ambientais:
          - **Temperatura (°C)**: Quanto maior a temperatura, maior a probabilidade de propagação do fogo.
          - **Umidade relativa (%)**: Quanto menor a umidade, maior a probabilidade de propagação do fogo.
          - **Densidade Vegetal (%)**: Quanto maior a densidade da vegetação, maior a probabilidade de propagação do fogo.
          - **Teor de umidade do combustível (%)**: Quanto menor a umidade do combustível, maior a probabilidade de propagação do fogo.
                ### Escala Técnica e Científica para Tipos de Vegetação na Propagação de Fogo
                Para criar uma escala metodológica tecnicocientífica para os tipos de vegetação na propagação de fogo, consideramos fatores como a inflamabilidade da vegetação, a densidade da biomassa, a capacidade de retenção de umidade, e a estrutura da vegetação. A escala a seguir é baseada em estudos científicos e dados empíricos sobre a propagação de incêndios florestais.
                
                #### Critérios de Avaliação
                1. **Inflamabilidade**: Capacidade da vegetação de pegar fogo rapidamente.
                2. **Densidade da Biomassa**: Quantidade de matéria orgânica presente na vegetação.
                3. **Retenção de Umidade**: Capacidade da vegetação de reter água e, portanto, resistir ao fogo.
                4. **Estrutura da Vegetação**: Complexidade estrutural da vegetação que pode afetar a propagação do fogo.
                
                #### Escala de Inflamabilidade da Vegetação
                
                ##### 1. Pastagem (0.4)
                - **Inflamabilidade**: Alta, devido à presença de gramíneas secas que pegam fogo rapidamente.
                - **Densidade da Biomassa**: Baixa a moderada, com predominância de gramíneas e ervas.
                - **Retenção de Umidade**: Baixa, gramíneas secam rapidamente e perdem umidade facilmente.
                - **Estrutura da Vegetação**: Simples, com pouca cobertura de dossel e vegetação rasteira.
                
                ##### 2. Matagal (0.6)
                - **Inflamabilidade**: Moderada a alta, devido à presença de arbustos e plantas herbáceas.
                - **Densidade da Biomassa**: Moderada, com presença de arbustos densos e alguma cobertura de solo.
                - **Retenção de Umidade**: Moderada, com plantas que podem reter alguma umidade, mas secam durante períodos de seca.
                - **Estrutura da Vegetação**: Moderadamente complexa, com vegetação densa que pode facilitar a propagação do fogo.
                
                ##### 3. Floresta Decídua (0.8)
                - **Inflamabilidade**: Moderada, devido à presença de folhas caídas que podem secar e pegar fogo.
                - **Densidade da Biomassa**: Alta, com grande quantidade de matéria orgânica, incluindo folhas, galhos e troncos.
                - **Retenção de Umidade**: Alta, devido à presença de árvores que podem reter umidade no solo e na vegetação.
                - **Estrutura da Vegetação**: Complexa, com várias camadas de vegetação (dossel, sub-bosque, camada de folhagem).
                
                ##### 4. Floresta Tropical (1.0)
                - **Inflamabilidade**: Baixa a moderada, devido à alta umidade, mas pode aumentar durante períodos de seca intensa.
                - **Densidade da Biomassa**: Muito alta, com grande quantidade de matéria orgânica em todas as camadas da floresta.
                - **Retenção de Umidade**: Muito alta, devido à alta precipitação e capacidade das plantas de reter umidade.
                - **Estrutura da Vegetação**: Muito complexa, com múltiplas camadas de vegetação densa, incluindo árvores altas, sub-bosque denso e camada de folhagem espessa.
                
                #### Justificativa Científica
                A escala acima é baseada em princípios ecológicos e estudos de campo que analisam a resposta da vegetação ao fogo. A inflamabilidade da vegetação é influenciada pela composição das espécies, a densidade da biomassa disponível para queima, a umidade retida na vegetação, e a estrutura física da vegetação que pode facilitar ou dificultar a propagação do fogo.
                
                - **Referências**:
                  1. Pyne, S. J., Andrews, P. L., & Laven, R. D. (1996). *Introduction to Wildland Fire*.
                  2. Bowman, D. M. J. S., Balch, J. K., Artaxo, P., Bond, W. J., Cochrane, M. A., D'Antonio, C. M., ... & Pyne, S. J. (2009). Fire in the Earth system. *Science*, 324(5926), 481-484.
                  3. Scott, J. H., & Reinhardt, E. D. (2001). *Assessing crown fire potential by linking models of surface and crown fire behavior*. USDA Forest Service, Rocky Mountain Research Station.
                
                Utilizando essa escala, podemos modelar a propagação do fogo de forma mais precisa, levando em consideração as características específicas de cada tipo de vegetação.
          - **Tipo de vegetação**: Diferentes tipos de vegetação têm diferentes probabilidades base de pegar fogo.
          - **Topografia (inclinação em graus)**: Áreas com maior inclinação podem ter maior probabilidade de propagação do fogo.
          - **Tipo de solo**: Diferentes tipos de solo influenciam a probabilidade de propagação do fogo.
                ### Escala Técnica e Científica para Tipos de Solo na Propagação de Fogo
                
                Para criar uma escala metodológica tecnicocientífica para os tipos de solo na propagação de fogo, consideramos fatores como a capacidade de retenção de umidade, a permeabilidade, e a influência do solo na vegetação. A escala a seguir é baseada em estudos científicos e dados empíricos sobre a influência do tipo de solo na propagação de incêndios.
                
                #### Critérios de Avaliação
                1. **Retenção de Umidade**: Capacidade do solo de reter água, influenciando a secagem da vegetação.
                2. **Permeabilidade**: Capacidade do solo de permitir a infiltração da água.
                3. **Influência na Vegetação**: O tipo de solo afeta o tipo e a densidade da vegetação que pode crescer, influenciando a inflamabilidade geral.
                
                #### Escala de Inflamabilidade do Solo
                
                ##### 1. Solo Arenoso (0.4)
                - **Retenção de Umidade**: Baixa, solo arenoso drena rapidamente a água, deixando a vegetação mais seca e suscetível a incêndios.
                - **Permeabilidade**: Alta, permite a rápida infiltração e drenagem da água.
                - **Influência na Vegetação**: Suporta vegetação menos densa e de raízes superficiais, que secam rapidamente.
                
                ##### 2. Solo Misto (0.6)
                - **Retenção de Umidade**: Moderada, retém mais água que o solo arenoso, mas ainda permite uma boa drenagem.
                - **Permeabilidade**: Moderada, combina características de solos arenosos e argilosos.
                - **Influência na Vegetação**: Suporta uma vegetação variada, com densidade moderada, o que influencia a propagação do fogo de forma intermediária.
                
                ##### 3. Solo Argiloso (0.8)
                - **Retenção de Umidade**: Alta, solo argiloso retém a água por mais tempo, mantendo a vegetação mais úmida.
                - **Permeabilidade**: Baixa, a água infiltra-se lentamente, o que pode levar a maior umidade do solo e da vegetação.
                - **Influência na Vegetação**: Suporta vegetação densa e de raízes profundas, que tende a ser menos inflamável devido à maior umidade.
                
                #### Justificativa Científica
                A escala acima é baseada em princípios pedológicos e estudos de campo que analisam a resposta do solo à secagem e à propagação de incêndios. A capacidade de retenção de umidade e a permeabilidade do solo influenciam diretamente a disponibilidade de água para a vegetação, afetando sua secagem e inflamabilidade.
                
                - **Referências**:
                  1. Brady, N. C., & Weil, R. R. (2008). *The Nature and Properties of Soils*. Pearson Education.
                  2. DeBano, L. F., Neary, D. G., & Ffolliott, P. F. (1998). *Fire's Effects on Ecosystems*. John Wiley & Sons.
                  3. Neary, D. G., Ryan, K. C., & DeBano, L. F. (2005). *Wildland Fire in Ecosystems: Effects of Fire on Soil and Water*. USDA Forest Service.
                
                Utilizando essa escala, podemos modelar a propagação do fogo de forma mais precisa, levando em consideração as características específicas de cada tipo de solo e sua influência na vegetação e na umidade do ambiente.
                
                Essa escala proporciona uma base técnica e científica para entender como diferentes tipos de solo afetam a propagação do fogo, permitindo simulações mais precisas e realistas.

        - **W_{effect}(i, j)**: Este fator é calculado com base na direção e velocidade do vento, influenciando a probabilidade de propagação do fogo na direção do vento:
          - **Velocidade do Vento (km/h)**: Quanto maior a velocidade do vento, maior a probabilidade de propagação do fogo.
          - **Direção do Vento (graus)**: A direção do vento influencia a direção preferencial de propagação do fogo.

        - **N_{effect}**: Este é um fator aleatório que introduz ruído na simulação, representando incertezas e variabilidades ambientais:
          - **Ruído (%)**: Define o nível de aleatoriedade na propagação do fogo, variando de 1% a 100%.

        ### Estatísticas e Interpretações
        A simulação permite observar como o fogo se propaga em diferentes condições ambientais. Os resultados podem ser utilizados para entender o comportamento do fogo e planejar estratégias de manejo e controle de incêndios.

        ### Análises Estatísticas
            
        #### Histogramas
        Um histograma é um gráfico de barras que mostra a frequência com que algo acontece. Imagine que estamos observando quantas árvores estão pegando fogo em uma floresta a cada dia durante um mês. No nosso simulador de fogo, um histograma nos ajuda a visualizar quantas células (pedaços de terreno) estão queimando em cada etapa do tempo da simulação. Por exemplo, se temos 10 células queimando no primeiro dia, 15 no segundo e assim por diante, o histograma mostrará essas contagens como barras, permitindo ver como o fogo se espalha ao longo do tempo.
        
        #### Gráficos de Margem de Erro
        Um gráfico de margem de erro mostra a média de um conjunto de dados e a variação ao redor dessa média. A média é como calcular a "nota média" de uma turma de alunos. No caso do nosso simulador, a média seria o número médio de células queimando a cada dia. A margem de erro indica o quanto esses números podem variar em torno da média. Se a margem de erro é pequena, significa que os números estão próximos da média; se é grande, significa que os números variam bastante. Isso ajuda a entender a consistência da propagação do fogo.
        
        #### Correlação de Spearman
        Correlação de Spearman é uma forma de medir a relação entre duas coisas sem assumir que essa relação seja linear. Por exemplo, podemos usar a correlação de Spearman para ver se existe uma relação entre a temperatura e a velocidade do fogo. Se a correlação for alta, significa que quando uma dessas variáveis aumenta, a outra também tende a aumentar (ou diminuir). Por exemplo, se a temperatura aumenta e o fogo se espalha mais rapidamente, isso indicaria uma alta correlação positiva.
        
        #### ANOVA
        ANOVA, ou Análise de Variância, é um teste estatístico que compara as médias de diferentes grupos para ver se há diferenças significativas entre eles. No contexto do simulador de fogo, podemos usar ANOVA para comparar diferentes cenários, como a propagação do fogo em diferentes tipos de vegetação (pastagem, matagal, floresta). Se os resultados mostram que há uma diferença significativa (um p-valor muito pequeno, como 1.0700732830108085e-21), significa que a propagação do fogo varia significativamente entre os tipos de vegetação.
        
        #### Q-Exponential
        A distribuição Q-Exponencial é uma maneira de modelar dados que não seguem uma distribuição normal. É útil quando estamos lidando com fenômenos complexos como a propagação do fogo, onde os eventos não acontecem de maneira previsível. Este método nos ajuda a entender melhor a distribuição dos dados em situações complexas. Por exemplo, os valores Q-Exponencial para o número de células queimando podem mostrar uma distribuição que não é simétrica, indicando que há dias em que o fogo se espalha muito mais rápido do que em outros.
        
        #### Estatística Q
        A Estatística Q é uma maneira de medir a relação entre variáveis em um contexto onde há dependência ou não linearidade entre elas. No contexto da propagação do fogo, a Estatística Q pode ajudar a entender como diferentes fatores (como temperatura, umidade, velocidade do vento) se combinam de maneiras complexas para influenciar a propagação do fogo. Em outras palavras, é uma ferramenta que nos ajuda a explorar e interpretar a complexidade dos dados que não seguem padrões simples.
        
        #### Matriz de Confusão
        Uma matriz de confusão é uma tabela usada para avaliar a performance de um modelo de classificação. No nosso simulador de fogo, a matriz de confusão mostra quantas vezes o modelo previu corretamente (ou incorretamente) o estado de uma célula (se estava intacta, queimando ou queimada). É como uma tabela que mostra quantas vezes um aluno acertou ou errou diferentes tipos de perguntas em uma prova. Por exemplo, se a matriz de confusão mostra que 243191 células foram corretamente identificadas como intactas e 126 células foram corretamente identificadas em cada estágio de queima, mas algumas previsões foram incorretas, isso nos ajuda a entender se o modelo está funcionando bem ou se precisa de ajustes.
        
        Exemplo de Matriz de Confusão:
          - [[243191, 129, 0, 0, 0, 0], [0, 0, 126, 0, 0, 0], [0, 0, 0, 126, 0, 0], [0, 0, 0, 0, 126, 0], [0, 0, 0, 0, 0, 126], [0, 0, 0, 0, 0, 6176]]
          - 243191 células foram corretamente identificadas como intactas.
          - 126 células foram corretamente identificadas em cada estágio de queima.
          - Alguns erros de previsão (129 células foram incorretamente identificadas como queimando quando não estavam).
        
        Essas análises estatísticas são importantes para entender melhor como o fogo se propaga e para melhorar a precisão do nosso simulador.
        """)

    # Definir parâmetros
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

    # Coleta dados históricos manuais
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

    # Converter dados históricos manuais para DataFrame
    df_historico_manual = pd.DataFrame(historico_manual)

    # Tamanho da grade e número de passos
    tamanho_grade = st.sidebar.slider('Tamanho da grade', 10, 100, 50)
    num_passos = st.sidebar.slider('Número de passos', 10, 200, 100)
    
    #_________________________________________________________________
    

    # Informações de contato
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)
    #_________________________________________________________________
    import streamlit as st
    import base64
    
    def main():
        # Adiciona um título na barra lateral
        st.sidebar.title("Controle de Áudio")
        
        # Lista de arquivos MP3
        mp3_files = {
            "Agente Alan Kay": "apresentação ac.mp3"
            
        }
    
        # Controle de seleção de música
        selected_mp3 = st.sidebar.radio("Escolha uma música", list(mp3_files.keys()))
    
        # Opção de loop
        loop = st.sidebar.checkbox("Repetir música")
    
        # Carregar e exibir o player de áudio
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
        #_____________________________________________________________________
    #_____________________________________________________________________


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
