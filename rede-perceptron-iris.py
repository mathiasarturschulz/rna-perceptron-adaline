import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from Class.Perceptron import Perceptron

# Dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

# Estrutura do dataset
print(df.head())

# Obtendo dados de duas classes: Iris-setosa e Iris-versicolor
# x e y são os sinais de entrada
# x => 100 primeiros registros, buscando as colunas 0 e 2
x = df.iloc[0:100, [0,2]].values
# y => 100 primeiros registros, onde: Iris-setosa = -1 e Iris-versicolor = 1
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Criação do modelo
perceptron = Perceptron (
    learning_rate = 0.1, 
    max_iterations = 10, 
    # random_weights = True, 
    stop_with_zero_error = True, 
)

# Treinamento
perceptron.train(x, y)

# Plot
print('Pesos Sinápticos (weights): %s' % perceptron.weights)


# Classificação (Apenas para gráficos 2D)
plot_decision_regions(x, y, clf = perceptron)
plt.title('Perceptron')
plt.xlabel('sepal comprimento [cm]')
plt.ylabel('petal comprimento [cm]')
plt.show()


# Tentativas e Qtd de erros
plt.plot(range(1, len(perceptron.errors)+1), perceptron.errors, marker = 'o')
plt.xlabel('Interações')
plt.ylabel('Classificação Incorretas')
plt.show()
