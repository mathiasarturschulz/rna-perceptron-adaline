import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from time import time


# Dataset
df = pd.read_excel('datasets/Dados_Treinamento_Adaline.xls', header = 0)
# Estrutura do dataset
print(df.head())

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values


# Variáveis
#
# x => x^(k) => Conjunto de amostras
# d => d^(k) => Saída desejada
# w => Pesos sinápticos (Valores aleatórios pequenos - entre 0 e 1 ?)
# n => Taxa de aprendizagem
# e => precisão requerida
# epoca => Número de épocas


class Adaline(object):
    
    # Construtor
    # LEARNING_RATE  => Taxa de aprendizado (entre 0 e 1)
    # PRECISION      => Precisão requerida
    # RANDOM_WEIGHTS => Pesos sinápticos iniciados com 0 ou aleatóriamente entre 0 e 1
    def __init__(
        self,
        learning_rate = 0.01,
        precision = 0.00001,
        random_weights = True,
    ):
        self.LEARNING_RATE = learning_rate
        self.PRECISION = precision
        self.RANDOM_WEIGHTS = random_weights

    # Método privado que gera o array de pesos sinápticos
    # Podem ser iniciados com 0 ou aleatóriamente entre 0 e 1
    # A posição 0 representará o bias
    def __initialWeights(self, x):
        number_weights = 1 + x.shape[1]
        if (self.RANDOM_WEIGHTS):
            return np.random.random_sample((number_weights,))
        return np.zeros(1 + x.shape[1])
    
    # Treinamento da Rede
    # Sinais de entrada: x e y
    # x => São as propriedades de cada registro
    # y => Determina a qual classe pertence
    def train(self, x, y):
        self.initialWeights = self.__initialWeights(x)
        self.w = list(self.initialWeights)
        
        self.epoca = 0

        eqm_anterior = 0
        eqm_atual = float("inf")

        print('iniciando loop...')
        # print(abs(eqm_atual - eqm_anterior) > self.PRECISION)
        # print(abs(eqm_atual - eqm_anterior))
        # print(eqm_atual - eqm_anterior)
        # print(self.PRECISION)
        while(abs(eqm_atual - eqm_anterior) > self.PRECISION):
            print('loop...')
            eqm_anterior = self.__eqm(x, y)

            for xi, yi in zip(x, y):
                u = self.LEARNING_RATE * (yi - self.__predict(xi))
                self.w[1:] += u * xi
                self.w[0] += u
            self.epoca = self.epoca + 1
            eqm_atual = self.__eqm(x, y)
        print('terminando loop...')
        return self


    def __eqm(self, x, y):
        p = len(x)
        eqm = 0

        for xi, yi in zip(x, y):
            u = self.LEARNING_RATE * (yi - self.__predict(xi))
            eqm = eqm + ((yi - u) * (yi - u))
        eqm = eqm / p
        print('eqm: %f' % eqm)
        return eqm


    # Validação da Rede
    # Sinais de entrada: x
    # x => São as propriedades de cada registro
    def validation(self, x):
        self.result_x = x
        self.result_y = np.zeros(0)

        # Passa por cada registro
        # xi => Propriedades do registro
        for xi in self.result_x:
            actualResult = self.__predict(xi)
            self.result_y = np.append(self.result_y, actualResult)
        return self

    # Método que calcula o potencial de ativação
    # Retorna a chance de acordo com os pesos de ser de ser uma determinada classe
    def __activationPotential(self, x):
        # self.weights[1:] => Pega o array, exceto o bias
        # Produto do array de propriedades com o array de pesos
        product = np.dot(x, self.w[1:])
        # Produto dos array mais o bias
        # Bias serve para aumentar o grau de liberdade dos ajustes dos pesos
        return product + self.w[0]
    
    # Método que cálcula a função de ativação
    # Retorna 1 se o novo peso for >= 0 ou -1 se o novo peso for < 0
    def __predict(self, x):
        return np.where(self.__activationPotential(x) >= 0.0, 1, -1)


# Criação do modelo
adaline = Adaline()

# Treinamento
adaline.train(x, y)
print('\n=> Treinamento: ')
print('Pesos Sinápticos - Weights (Obs: Posição 0 é o bias):')
print('Iniciais => %s' % adaline.initialWeights)
print('Finais   => %s' % adaline.w)
print('Número de Épocas => %s' % adaline.epoca)
print('Taxa de Aprendizado => %s' % adaline.LEARNING_RATE)
print('Precisão requerida => %s' % adaline.PRECISION)

# Validação
# Dataset
df = pd.read_excel('datasets/Dados_Validação_Adaline.xls', header = 0)

# Obtenção dos dados
x = df.iloc[:, [0, 1, 2, 3]].values

print('\n=> Validação: ')
adaline.validation(x)
print('X - Propriedades de cada registro: ')
print(adaline.result_x)
print('Y - Classe de cada registro: ')
print(adaline.result_y)

print('terminou...')