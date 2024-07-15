import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

# Inicializa a matriz do autômato celular
def initialize_grid(size, fire_start):
    grid = np.zeros((size, size), dtype=int)  # Cria uma matriz de zeros (células vivas)
    grid[fire_start] = BURNING1  # Define a célula inicial como queimando
    return grid

# Aplica a regra do autômato celular
def apply_fire_rules(grid, wind_direction):
    new_grid = grid.copy()  # Cria uma cópia da matriz para atualizar os estados
    size = grid.shape[0]  # Obtém o tamanho da matriz

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
                if grid[i-1, j] == ALIVE and np.random.rand() < probabilities[ALIVE] * wind_effect(wind_direction, 'N'):
                    new_grid[i-1, j] = BURNING1
                if grid[i+1, j] == ALIVE and np.random.rand() < probabilities[ALIVE] * wind_effect(wind_direction, 'S'):
                    new_grid[i+1, j] = BURNING1
                if grid[i, j-1] == ALIVE and np.random.rand() < probabilities[ALIVE] * wind_effect(wind_direction, 'W'):
                    new_grid[i, j-1] = BURNING1
                if grid[i, j+1] == ALIVE and np.random.rand() < probabilities[ALIVE] * wind_effect(wind_direction, 'E'):
                    new_grid[i, j+1] = BURNING1
    return new_grid

# Função para modelar o efeito do vento
def wind_effect(wind_direction, direction):
    effect = 1.0  # Efeito padrão (sem alteração)
    if wind_direction == direction:
        effect = 1.5  # Aumenta a probabilidade se o vento estiver na mesma direção
    elif (wind_direction == 'N' and direction == 'S') or (wind_direction == 'S' and direction == 'N') or \
         (wind_direction == 'E' and direction == 'W') or (wind_direction == 'W' and direction == 'E'):
        effect = 0.5  # Reduz a probabilidade se o vento estiver na direção oposta
    return effect

# Função para executar a simulação
def run_simulation(size, steps, fire_start, wind_direction):
    grid = initialize_grid(size, fire_start)  # Inicializa a matriz do autômato celular
    grids = [grid.copy()]  # Cria uma lista para armazenar os estados em cada passo

    for _ in range(steps):  # Executa a simulação para o número de passos definido
        grid = apply_fire_rules(grid, wind_direction)  # Aplica as regras do autômato
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
            if wind_direction == 'E':
                ax.arrow(80, 90, 10, 0, head_width=5, head_length=5, fc='blue', ec='blue')
                ax.text(75, 120, 'Vento Leste', color='blue', fontsize=12)
            elif wind_direction == 'W':
                ax.arrow(20, 90, -10, 0, head_width=5, head_length=5, fc='blue', ec='blue')
                ax.text(15, 95, 'Vento Oeste', color='blue', fontsize=12)
            elif wind_direction == 'N':
                ax.arrow(90, 80, 0, -10, head_width=5, head_length=5, fc='blue', ec='blue')
                ax.text(95, 85, 'Vento Norte', color='blue', fontsize=12)
            elif wind_direction == 'S':
                ax.arrow(90, 20, 0, 10, head_width=5, head_length=5, fc='blue', ec='blue')
                ax.text(95, 25, 'Vento Sul', color='blue', fontsize=12)

        ax.grid(True)  # Exibe a malha cartesiana

    # Cria a legenda para os diferentes estados das células
    handles = [plt.Rectangle((0,0),1,1, color=cmap.colors[i]) for i in range(6)]
    labels = ['Intacto', 'Queimando1', 'Queimando2', 'Queimando3', 'Queimando4', 'Queimado']
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    st.pyplot(fig)

# Interface do Streamlit
st.title("EcoSim.ai - Simulador de Propagação de Incêndio em Autômatos Celulares")

# Definir parâmetros
params = {
    'temperature': st.slider('Temperatura (°C)', 0, 50, 30),
    'humidity': st.slider('Umidade relativa (%)', 0, 100, 40),
    'wind_speed': st.slider('Velocidade do Vento (km/h)', 0, 100, 20),
    'wind_direction': st.selectbox('Direção do Vento', ['N', 'S', 'E', 'W']),
    'precipitation': st.slider('Precipitação (mm/dia)', 0, 200, 0),
    'solar_radiation': st.slider('Radiação Solar (W/m²)', 0, 1200, 800),
    'vegetation_type': st.selectbox('Tipo de vegetação', ['pastagem', 'matagal', 'floresta']),
    'vegetation_density': st.slider('Densidade Vegetal (%)', 0, 100, 70),
    'fuel_moisture': st.slider('Teor de umidade do combustível (%)', 0, 100, 10),
    'topography': st.slider('Topografia (inclinação em graus)', 0, 45, 5),
    'soil_type': st.selectbox('Tipo de solo', ['arenoso', 'argiloso', 'argiloso']),
    'ndvi': st.slider('NDVI (Índice de Vegetação por Diferença Normalizada)', 0.0, 1.0, 0.6),
    'fire_intensity': st.slider('Intensidade do Fogo (kW/m)', 0, 10000, 5000),
    'time_since_last_fire': st.slider('Tempo desde o último incêndio (anos)', 0, 100, 10),
    'human_intervention': st.slider('Fator de Intervenção Humana (escala 0-1)', 0.0, 1.0, 0.2)
}

# Tamanho da grade e número de passos
grid_size = st.slider('Tamanho da grade', 10, 100, 50)
num_steps = st.slider('Número de passos', 10, 200, 100)

if st.button('Executar Simulação'):
    fire_start = (grid_size // 2, grid_size // 2)
    wind_direction = params['wind_direction']
    simulation = run_simulation(grid_size, num_steps, fire_start, wind_direction)
    plot_simulation(simulation, fire_start, wind_direction)
