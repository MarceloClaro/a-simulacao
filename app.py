import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd

# Define estados das células
ALIVE = 0        # Célula viva (verde)
BURNING1 = 1     # Célula começando a queimar (amarelo)
BURNING2 = 2     # Célula continuando a queimar (laranja)
BURNING3 = 3     # Célula continuando a queimar (vermelho)
BURNING4 = 4     # Célula continuando a queimar (vermelho escuro)
BURNED = 5       # Célula queimada (preto)

# Define as probabilidades de propagação do fogo para cada estado
probabilities = {
    ALIVE: 0.6,       # Probabilidade de uma célula viva pegar fogo
    BURNING1: 0.8,    # Probabilidade de uma célula queimando continuar queimando
    BURNING2: 0.8,    # Continuação da queima
    BURNING3: 0.8,    # Continuação da queima
    BURNING4: 0.8,    # Continuação da queima
    BURNED: 0         # Uma célula queimada não pode pegar fogo novamente
}

# Atribui valores numéricos ao tipo de vegetação
vegetation_type_values = {
    'pastagem': 0.5,
    'matagal': 0.75,
    'floresta': 1.0
}

# Inicializa a matriz do autômato celular
def initialize_grid(size, fire_start):
    grid = np.zeros((size, size), dtype=int)  # Cria uma matriz de zeros (células vivas)
    grid[fire_start] = BURNING1  # Define a célula inicial como queimando
    return grid

# Função para calcular a probabilidade de propagação com base nos parâmetros
def calculate_spread_probability(params):
    temp_factor = (params['temperature'] - 20) / 30
    humidity_factor = (100 - params['humidity']) / 100
    wind_speed_factor = params['wind_speed'] / 50
    vegetation_density_factor = params['vegetation_density'] / 100
    fuel_moisture_factor = (100 - params['fuel_moisture']) / 100
    topography_factor = params['topography'] / 45
    ndvi_factor = params['ndvi']
    fire_intensity_factor = params['fire_intensity'] / 10000
    human_intervention_factor = 1 - params['human_intervention']
    vegetation_type_factor = vegetation_type_values[params['vegetation_type']]

    base_prob = 0.3
    prob = base_prob + 0.1 * (temp_factor + humidity_factor + wind_speed_factor + vegetation_density_factor +
                              fuel_moisture_factor + topography_factor + ndvi_factor + fire_intensity_factor +
                              vegetation_type_factor) * human_intervention_factor

    return min(max(prob, 0), 1)

# Aplica a regra do autômato celular
def apply_fire_rules(grid, params, noise):
    new_grid = grid.copy()  # Cria uma cópia da matriz para atualizar os estados
    size = grid.shape[0]  # Obtém o tamanho da matriz
    spread_prob = calculate_spread_probability(params)

    for i in range(1, size - 1):  # Percorre cada célula (ignorando bordas)
        for j in range(1, size - 1):
            if grid[i, j] == BURNING1:
                new_grid[i, j] = BURNING2  # Atualiza célula para o próximo estado de queima
            elif grid[i, j] == BURNING2:
                new_grid[i, j] = BURNING3
            elif grid[i, j] == BURNING3:
                new_grid[i, j] = BURNING4
            elif grid[i, j] == BURNING4:
                new_grid[i, j] = BURNED
                # Propaga o fogo para células adjacentes com base na probabilidade e efeito do vento
                if grid[i-1, j] == ALIVE and np.random.rand() < spread_prob * wind_effect(params['wind_direction'], (i-1, j), (i, j)) * noise_effect(noise):
                    new_grid[i-1, j] = BURNING1
                if grid[i+1, j] == ALIVE and np.random.rand() < spread_prob * wind_effect(params['wind_direction'], (i+1, j), (i, j)) * noise_effect(noise):
                    new_grid[i+1, j] = BURNING1
                if grid[i, j-1] == ALIVE and np.random.rand() < spread_prob * wind_effect(params['wind_direction'], (i, j-1), (i, j)) * noise_effect(noise):
                    new_grid[i, j-1] = BURNING1
                if grid[i, j+1] == ALIVE and np.random.rand() < spread_prob * wind_effect(params['wind_direction'], (i, j+1), (i, j)) * noise_effect(noise):
                    new_grid[i, j+1] = BURNING1
    return new_grid

# Função para modelar o efeito do vento
def wind_effect(wind_direction, cell, source):
    wind_angle_rad = np.deg2rad(wind_direction)
    wind_vector = np.array([np.cos(wind_angle_rad), np.sin(wind_angle_rad)])
    
    direction_vector = np.array([cell[0] - source[0], cell[1] - source[1]])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    effect = np.dot(wind_vector, direction_vector)
    effect = (effect + 1) / 2  # Normaliza para um valor entre 0 e 1
    
    return effect

# Função para modelar o efeito do ruído
def noise_effect(noise):
    return 1 + (np.random.rand() - 0.5) * (noise / 50.0)

# Função para executar a simulação
def run_simulation(size, steps, fire_start, params, noise):
    grid = initialize_grid(size, fire_start)  # Inicializa a matriz do autômato celular
    grids = [grid.copy()]  # Cria uma lista para armazenar os estados em cada passo

    for _ in range(steps):  # Executa a simulação para o número de passos definido
        grid = apply_fire_rules(grid, params, noise)  # Aplica as regras do autômato
        grids.append(grid.copy())  # Armazena a matriz atualizada na lista

    return grids

# Função para plotar a simulação
def plot_simulation(simulation, fire_start, wind_direction):
    num_plots = min(50, len(simulation))  # Define o número máximo de gráficos a serem plotados
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))  # Cria um grid de subplots (5 linhas, 10 colunas)
    axes = axes.flatten()  # Achata a matriz de eixos para fácil iteração

    # Define um mapa de cores personalizado para os diferentes estados das células
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red', 'darkred', 'black'])

    # Itera sobre os estados da simulação para plotar cada um
    for i, grid in enumerate(simulation[::max(1, len(simulation)//num_plots)]):
        if i >= len(axes):  # Verifica se o número máximo de gráficos foi atingido
            break
        ax = axes[i]
        ax.imshow(grid, cmap=cmap, interpolation='nearest')  # Plota a matriz atual com o mapa de cores
        ax.set_title(f'Passo {i * (len(simulation)//num_plots)}')  # Define o título do subplot com o passo da simulação

        # Marca o quadrinho inicial com um quadrado vermelho
        if i == 0:
            ax.plot(fire_start[1], fire_start[0], 'rs', markersize=5, label='Fogo Inicial')
            ax.legend(loc='upper right')

        # Desenha uma seta para indicar a direção do vento com texto
        if i == len(axes) - 1:  # Último gráfico
            ax.arrow(90, 90, 10 * np.cos(np.deg2rad(wind_direction)), 10 * np.sin(np.deg2rad(wind_direction)),
                     head_width=5, head_length=5, fc='blue', ec='blue')
            ax.text(80, 95, f'Vento {wind_direction}°', color='blue', fontsize=12)

        ax.grid(True)  # Exibe a malha cartesiana

    # Cria a legenda para os diferentes estados das células
    handles = [plt.Rectangle((0,0),1,1, color=cmap.colors[i]) for i in range(6)]
    labels = ['Intacto', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado']
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    st.pyplot(fig)

# Função para plotar histogramas e gráficos de margem de erro
def plot_histograms_and_errors(simulation):
    burn_counts = [np.sum(grid == BURNING1) + np.sum(grid == BURNING2) + np.sum(grid == BURNING3) + np.sum(grid == BURNING4) for grid in simulation]
    burn_counts_df = pd.DataFrame(burn_counts, columns=["Burning Cells"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma
    sns.histplot(burn_counts_df, x="Burning Cells", ax=ax[0], kde=True, bins=20, color='orange')
    ax[0].set_title('Histograma de Células Queimando')
    ax[0].set_xlabel('Número de Células Queimando')
    ax[0].set_ylabel('Frequência')
    
    # Gráfico de média e margem de erro
    rolling_mean = burn_counts_df.rolling(window=10).mean()
    rolling_std = burn_counts_df.rolling(window=10).std()
    ax[1].plot(rolling_mean, label='Média', color='blue')
    ax[1].fill_between(rolling_std.index, rolling_mean["Burning Cells"] - rolling_std["Burning Cells"], rolling_mean["Burning Cells"] + rolling_std["Burning Cells"], color='blue', alpha=0.2, label='Margem de Erro (1 std)')
    ax[1].set_title('Média e Margem de Erro')
    ax[1].set_xlabel('Passos da Simulação')
    ax[1].set_ylabel('Número de Células Queimando')
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Função para calcular correlações e realizar ANOVA, Q-Exponential e matriz de confusão
def perform_advanced_statistics(simulation, params):
    burn_counts = [np.sum(grid == BURNING1) + np.sum(grid == BURNING2) + np.sum(grid == BURNING3) + np.sum(grid == BURNING4) for grid in simulation]
    burn_counts_df = pd.DataFrame(burn_counts, columns=["Burning Cells"])

    # Adiciona os parâmetros à análise
    param_values = pd.DataFrame([params] * len(burn_counts_df))
    param_values['Burning Cells'] = burn_counts_df['Burning Cells']

    # Correlação de Spearman
    spearman_corr = param_values.corr(method='spearman')
    st.write("### Matriz de Correlação (Spearman):")
    st.write(spearman_corr)

    # ANOVA
    thirds = np.array_split(burn_counts_df["Burning Cells"], 3)
    f_val, p_val = stats.f_oneway(thirds[0], thirds[1], thirds[2])
    st.write("### Resultado da ANOVA:")
    st.write(f"F-valor: {f_val}, p-valor: {p_val}")

    # Q-Exponential
    def q_exponential(values, q):
        return (1 - (1 - q) * values)**(1 / (1 - q))

    q_value = 1.5  # Exemplo de valor de q
    q_exponential_values = q_exponential(burn_counts_df["Burning Cells"], q_value)
    st.write("### Valores Q-Exponential:")
    st.write(q_exponential_values)

    # Matriz de Confusão
    y_true = np.concatenate([grid.flatten() for grid in simulation[:-1]])
    y_pred = np.concatenate([grid.flatten() for grid in simulation[1:]])
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[ALIVE, BURNING1, BURNING2, BURNING3, BURNING4, BURNED])
    st.write("### Matriz de Confusão:")
    st.write(conf_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ALIVE', 'BURNING1', 'BURNING2', 'BURNING3', 'BURNING4', 'BURNED'], yticklabels=['ALIVE', 'BURNING1', 'BURNING2', 'BURNING3', 'BURNING4', 'BURNED'])
    ax.set_xlabel('Estado Previsto')
    ax.set_ylabel('Estado Real')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

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
          - **Tipo de vegetação**: Diferentes tipos de vegetação têm diferentes probabilidades base de pegar fogo.
          - **Topografia (inclinação em graus)**: Áreas com maior inclinação podem ter maior probabilidade de propagação do fogo.

        - **W_{effect}(i, j)**: Este fator é calculado com base na direção e velocidade do vento, influenciando a probabilidade de propagação do fogo na direção do vento:
          - **Velocidade do Vento (km/h)**: Quanto maior a velocidade do vento, maior a probabilidade de propagação do fogo.
          - **Direção do Vento (graus)**: A direção do vento influencia a direção preferencial de propagação do fogo.

        - **N_{effect}**: Este é um fator aleatório que introduz ruído na simulação, representando incertezas e variabilidades ambientais:
          - **Ruído (%)**: Define o nível de aleatoriedade na propagação do fogo, variando de 1% a 100%.

        ### Estatísticas e Interpretações
        A simulação permite observar como o fogo se propaga em diferentes condições ambientais. Os resultados podem ser utilizados para entender o comportamento do fogo e planejar estratégias de manejo e controle de incêndios.

        ### Análises Estatísticas
        - **Histogramas**: Visualização da distribuição das células queimando ao longo do tempo.
        - **Gráficos de Margem de Erro**: Visualização da média e margem de erro da propagação do fogo.
        - **Correlação de Spearman**: Medida não paramétrica da dependência entre duas variáveis.
        - **ANOVA**: Análise de variância para verificar diferenças significativas entre os grupos.
        - **Q-Exponential**: Distribuição Q-Exponencial para modelagem de dados.
        - **Matriz de Confusão**: Ferramenta para medir o desempenho do modelo de classificação.
        """)

    # Definir parâmetros
    params = {
        'temperature': st.sidebar.slider('Temperatura (°C)', 0, 50, 30),
        'humidity': st.sidebar.slider('Umidade relativa (%)', 0, 100, 40),
        'wind_speed': st.sidebar.slider('Velocidade do Vento (km/h)', 0, 100, 20),
        'wind_direction': st.sidebar.slider('Direção do Vento (graus)', 0, 360, 90),
        'precipitation': st.sidebar.slider('Precipitação (mm/dia)', 0, 200, 0),
        'solar_radiation': st.sidebar.slider('Radiação Solar (W/m²)', 0, 1200, 800),
        'vegetation_type': st.sidebar.selectbox('Tipo de vegetação', ['pastagem', 'matagal', 'floresta']),
        'vegetation_density': st.sidebar.slider('Densidade Vegetal (%)', 0, 100, 70),
        'fuel_moisture': st.sidebar.slider('Teor de umidade do combustível (%)', 0, 100, 10),
        'topography': st.sidebar.slider('Topografia (inclinação em graus)', 0, 45, 5),
        'soil_type': st.sidebar.selectbox('Tipo de solo', ['arenoso', 'argiloso', 'argiloso']),
        'ndvi': st.sidebar.slider('NDVI (Índice de Vegetação por Diferença Normalizada)', 0.0, 1.0, 0.6),
        'fire_intensity': st.sidebar.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
        'time_since_last_fire': st.sidebar.slider('Tempo desde o último incêndio (anos)', 0, 100, 10),
        'human_intervention': st.sidebar.slider('Fator de Intervenção Humana (escala 0-1)', 0.0, 1.0, 0.2),
        'noise': st.sidebar.slider('Ruído (%)', 1, 100, 10)
    }

    # Tamanho da grade e número de passos
    grid_size = st.sidebar.slider('Tamanho da grade', 10, 100, 50)
    num_steps = st.sidebar.slider('Número de passos', 10, 200, 100)

    if st.button('Executar Simulação'):
        fire_start = (grid_size // 2, grid_size // 2)
        noise = params['noise']
        simulation = run_simulation(grid_size, num_steps, fire_start, params, noise)
        plot_simulation(simulation, fire_start, params['wind_direction'])
        plot_histograms_and_errors(simulation)
        perform_advanced_statistics(simulation, params)

if __name__ == "__main__":
    main()
