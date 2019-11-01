import numpy as np
import pandas as pd

# Adaline
class Adaline(object):

    # Construtor
    # LEARNING_RATE  => Taxa de aprendizado (entre 0 e 1)
    # PRECISION      => Precisão requerida
    # RANDOM_WEIGHTS => Pesos sinápticos iniciados com 0 ou aleatóriamente entre 0 e 1
    def __init__(
        self,
        learning_rate = 0.01,
        precision = 0.00001,
        random_weights = False,
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
        self.weights = list(self.initialWeights)
        self.iterations = 0

        # Erro Quadrático Médio
        eqm_anterior = 0
        eqm_atual = float("inf")

        # Repete enquanto o erro foi maior que a precisão mínima estabelecida
        while(abs(eqm_atual - eqm_anterior) > self.PRECISION):
            eqm_anterior = self.__eqm(x, y)

            for xi, yi in zip(x, y):
                # Potencial de ativação
                u = self.__activationPotential(xi)

                # Cálculo dos novos pesos de acordo com o potencial de ativação
                self.weights[1:] += self.LEARNING_RATE * (yi - u) * xi
                self.weights[0] += self.LEARNING_RATE * (yi - u)

            self.iterations += 1
            eqm_atual = self.__eqm(x, y)
        return self

    # Método que cálcula o EQM (Erro Quadrático Médio)
    # Média da diferença entre o valor do estimador e do parâmetro ao quadrado
    def __eqm(self, x, y):
        # Quantidade de registros
        p = len(x)
        eqm = 0

        # Percorre cada registro e sua classe
        for xi, yi in zip(x, y):
            u = self.__activationPotential(xi)
            eqm += ((yi - u))**2
        eqm = eqm / p
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
            actualResult = self.predict(xi)
            self.result_y = np.append(self.result_y, actualResult)
        return self

    # Método que calcula o potencial de ativação
    # Retorna a chance de acordo com os pesos de ser de ser uma determinada classe
    def __activationPotential(self, x):
        # self.weights[1:] => Pega o array, exceto o bias
        # Produto do array de propriedades com o array de pesos
        product = np.dot(x, self.weights[1:])
        # Produto dos array mais o bias
        # Bias serve para aumentar o grau de liberdade dos ajustes dos pesos
        return product + self.weights[0]
    
    # Método que cálcula a função de ativação
    # Retorna 1 se o novo peso for >= 0 ou -1 se o novo peso for < 0
    def predict(self, x):
        return np.where(self.__activationPotential(x) >= 0.0, 1, -1)
