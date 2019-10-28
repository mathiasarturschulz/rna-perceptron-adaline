# Módulos necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from time import time
# %matplotlib inline


# Algoritmo Perceptron
# eta => Taxa de aprendizagem (learning rate) - valores entre 0 e 1
# epoch => Número de passos / tentativas no dataset de treino
class Perceptron(object):
    
    # Construtor
    def __init__(self, eta = 0.01, epochs = 50):
        self.eta = eta
        self.epochs = epochs
    
    # Método de treinamento
    def train(self, x, y):
        
        # w => Pesos sinápticos
        # x.shape[1] => Retorna a dimenção do array
        # w inicialmente recebe zeros
        self.w = np.zeros(1 + x.shape[1])
        self.errors_ = []
        
        # Repetirá o número de tentativas especificadas
        for _ in range(self.epochs):
            # errors => Quantidade de erros
            errors = 0

            # Passa por todos os registros
            for xi, target in zip(x, y):
                update = self.eta * (target - self. predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    # Calcula um novo peso
    def net_input(self, x):
        # np.dot => Produto dos arrays
        # self.w[1:] => Pega o array (exceto posição 0)
        # np.dot(x, self.w[1:]) => Produto dos array
        # Retorna com o peso sináptico da posição 0
        return np.dot(x, self.w[1:]) + self.w[0]
    
    # Retorna 1 se o novo peso for >= 0 ou -1 se o novo peso for < 0
    def predict(self, x):
        return np.where(self.net_input(x)>= 0.0, 1, -1)


# Dataset Iris
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
# eta => Taxa de aprendizagem (learning rate) - valores entre 0 e 1
# epoch => Número de passos / tentativas no dataset de treino
clf_perceptron = Perceptron(eta = 0.1, epochs = 10)


# Treinamento
clf_perceptron.train(x, y)


# Plot
print('Pesos Sinápticos (weights): %s' % clf_perceptron.w)


# Classificação
plot_decision_regions(x, y, clf = clf_perceptron)
plt.title('Perceptron')
plt.xlabel('sepal comprimento [cm]')
plt.ylabel('petal comprimento [cm]')
plt.show()


# Tentativas e Qtd de erros
plt.plot(range(1, len(clf_perceptron.errors_)+1), clf_perceptron.errors_, marker = 'o')
plt.xlabel('Interações')
plt.ylabel('Classificação Incorretas')
plt.show()
