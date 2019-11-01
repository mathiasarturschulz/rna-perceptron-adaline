import numpy as np
import pandas as pd

# Perceptron
class Perceptron(object):
    
    # Construtor
    # LEARNING_RATE        => Taxa de aprendizado (entre 0 e 1)
    # MAX_ITERATIONS       => Número de tentativas no dataset de treino
    # RANDOM_WEIGHTS       => Pesos sinápticos iniciados com 0 ou aleatóriamente entre 0 e 1
    # STOP_WITH_ZERO_ERROR => Determina se deve parar quando a predição foi correta
    #                         ou se deve parar somente no MAX_ITERATIONS
    def __init__(
        self, 
        learning_rate = 0.01, 
        max_iterations = 50, 
        random_weights = False,
        stop_with_zero_error = False
    ):
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.RANDOM_WEIGHTS = random_weights
        self.STOP_WITH_ZERO_ERROR = stop_with_zero_error

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
        self.iterations = self.MAX_ITERATIONS
        self.errors = []
        
        # Repetirá o número de tentativas especificadas
        for actualIteration in range(self.MAX_ITERATIONS):
            actualErrors = 0

            # Passa por cada registro
            # xi => Propriedades do registro
            # yi => Classe do registro
            for xi, yi in zip(x, y):
                # Quando acerta:
                #     classe - predição = 0
                #     Os pesos não serão alterados 
                # Taxa de aprendizado * (classe - predição)
                update = self.LEARNING_RATE * (yi - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                actualErrors += int(update != 0.0)
            self.errors.append(actualErrors)
            if (self.STOP_WITH_ZERO_ERROR and actualErrors == 0):
                self.iterations = actualIteration
                break
        return self
    
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
