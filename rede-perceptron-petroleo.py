import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from time import time
from Class.Perceptron import Perceptron

# Dataset
df = pd.read_excel('datasets/Dados_Treinamento_Perceptron.xls', header = 0)
# Estrutura do dataset
print(df.head())

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, 3].values

# Criação do modelo
perceptron = Perceptron (
    learning_rate = 0.1, 
    max_iterations = 500, 
    # random_weights = True, 
    stop_with_zero_error = True, 
)


# Treinamento
perceptron.train(x, y)
print('\n=> Treinamento: ')
print('Pesos Sinápticos - Weights (Obs: Posição 0 é o bias):')
print('Iniciais => %s' % perceptron.initialWeights)
print('Finais   => %s' % perceptron.weights)
print('Número de Épocas => %s' % perceptron.iterations)
print('QTD de erros da época %s => %s' % (perceptron.iterations, perceptron.errors[-1]))
print('Taxa de Aprendizado => %s' % perceptron.LEARNING_RATE)

# Tentativas e Qtd de erros
plt.plot(range(1, len(perceptron.errors)+1), perceptron.errors, marker = 'o')
plt.title('Quantidade de previsões incorretas de acordo com as épocas')
plt.xlabel('Iterações')
plt.ylabel('Classificações incorretas')
plt.show()


# Validação
# Dataset
df = pd.read_excel('datasets/Dados_Validação_Perceptron.xls', header = 0)

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2]].values

print('\n=> Validação: ')
perceptron.validation(x)
print('X - Propriedades de cada registro: ')
print(perceptron.result_x)
print('Y - Classe de cada registro: ')
print(perceptron.result_y)
