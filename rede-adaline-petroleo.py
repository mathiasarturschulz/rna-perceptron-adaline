import numpy as np
import pandas as pd
from Class.Adaline import Adaline

# Dataset
df = pd.read_excel('datasets/Dados_Treinamento_Perceptron.xls', header = 0)
# Estrutura do dataset
print(df.head())

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, 3].values

# Criação do modelo
adaline = Adaline(
    learning_rate = 0.01,
    precision = 0.00001,
    # random_weights = True,
)


# Treinamento
adaline.train(x, y)
print('\n=> Treinamento: ')
print('Pesos Sinápticos - Weights (Obs: Posição 0 é o bias):')
print('Iniciais => %s' % adaline.initialWeights)
print('Finais   => %s' % adaline.weights)
print('Taxa de Aprendizado => %s' % adaline.LEARNING_RATE)
print('Número de Épocas => %s' % adaline.iterations)
print('Precisão requerida => %s' % adaline.PRECISION)


# Validação
# Dataset
df = pd.read_excel('datasets/Dados_Validação_Perceptron.xls', header = 0)

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2]].values

print('\n=> Validação: ')
adaline.validation(x)
print('X - Propriedades de cada registro: ')
print(adaline.result_x)
print('Y - Classe de cada registro: ')
print(adaline.result_y)
