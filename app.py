import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Classe para simulação da propagação de incêndio
class FirePropagationSimulator:
    def __init__(self, grid_size, num_steps, params):
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.params = params
        self.grid = np.zeros((grid_size, grid_size))
        
    def initialize_fire(self):
        center = self.grid_size // 2
        self.grid[center, center] = 1
        
    def calculate_spread_probability(self, i, j):
        base_prob = 0.3
        temp_factor = (self.params['temperature'] - 20) / 40  # Efeito de temperatura normalizado
        humidity_factor = (100 - self.params['humidity']) / 100
        wind_factor = self.params['wind_speed'] / 50
        vegetation_factor = self.params['vegetation_density'] / 100
        ndvi_factor = self.params['ndvi']
        prob = base_prob + 0.1 * (temp_factor + humidity_factor + wind_factor + vegetation_factor + ndvi_factor)
        return min(max(prob, 0), 1)
    
    def update_cell(self, i, j):
        if self.grid[i, j] == 1:  # Se a célula estiver queimando
            self.grid[i, j] = 2  # Define como queimado
        elif self.grid[i, j] == 0:  # Se a célula não estiver queimada
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ni, nj in neighbors:
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    if self.grid[ni, nj] == 1:  # Se o vizinho estiver queimando
                        if random.random() < self.calculate_spread_probability(i, j):
                            self.grid[i, j] = 1  # Coloca fogo na célula
                            break
    
    def step(self):
        new_grid = self.grid.copy()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.update_cell(i, j)
        self.grid = new_grid

    def run_simulation(self):
        self.initialize_fire()
        fire_progression = [self.grid.copy()]
        
        for _ in range(self.num_steps):
            self.step()
            fire_progression.append(self.grid.copy())
            
        return fire_progression
    
    def plot_results(self, fire_progression):
        cmap = ListedColormap(['green', 'red', 'black'])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, grid in enumerate(fire_progression[::len(fire_progression)//5]):
            if i < 6:
                axes[i].imshow(grid, cmap=cmap)
                axes[i].set_title(f'Step {i * (len(fire_progression)//5)}')
                axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)

# Interface do Streamlit
st.title("EcoSim.ai - Simulador de Propagação de Incêndio em Autômatos Celulares")

# Definir parâmetros
params = {
    'temperature': st.slider('Temperatura (°C)', 0, 50, 30),
    'humidity': st.slider('Umidade relativa (%)', 0, 100, 40),
    'wind_speed': st.slider('Velocidade do Vento (km/h)', 0, 100, 20),
    'wind_direction': st.slider('Direção do Vento (graus)', 0, 360, 90),
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
    simulator = FirePropagationSimulator(grid_size=grid_size, num_steps=num_steps, params=params)
    fire_progression = simulator.run_simulation()
    simulator.plot_results(fire_progression)
