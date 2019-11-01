import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Class.Adaline import Adaline

# Dataset
df = pd.read_excel('datasets/Dados_Treinamento_Adaline.xls', header = 0)
# Estrutura do dataset
print(df.head())

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values

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

# Validação com os mesmos registros de treinamento
print('\n=> Validação: ')
print('Usando os registro do Dados_Treinamento_Adaline.xls')
print('Os mesmos usados para treinamento')
adaline.validation(x)
# print('X - Propriedades de cada registro: ')
# print(adaline.result_x)
print('Classe esperada: ')
print(y)
print('Classe obtida da validação: ')
print(adaline.result_y)

erro = 0
for y1, y2 in zip(y, adaline.result_y):
    if y1 != y2:
        erro += 1
print('Quantidade de classes diferente do esperado: %i' % erro)


# Gráfico: Épocas X EQM
plt.plot(range(1, adaline.iterations+1), adaline.eqm)
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')
plt.show()


# # Validação
# # Dataset
# df = pd.read_excel('datasets/Dados_Validação_Adaline.xls', header = 0)

# # Obtenção dos dados
# x = df.iloc[:, [0, 1, 2, 3]].values

# print('\n=> Validação: ')
# adaline.validation(x)
# print('X - Propriedades de cada registro: ')
# print(adaline.result_x)
# print('Y - Classe de cada registro: ')
# print(adaline.result_y)
