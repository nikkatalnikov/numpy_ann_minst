import numpy as np
from ann import helpers


class ANN:
    def __init__(self, x_dim: int, y_dim: int, layers_dim: int, Lambda: float, learning_rate: float):
        self.Lambda = Lambda
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.layers_dim = layers_dim
        self.learning_rate = learning_rate
        self.b1 = np.zeros(self.layers_dim)
        self.b2 = np.zeros(self.layers_dim)
        self.b3 = np.zeros(y_dim)
        self.W1 = np.random.rand(x_dim, self.layers_dim) * np.sqrt(1 / x_dim)
        self.W2 = np.random.rand(self.layers_dim, self.layers_dim) * np.sqrt(1 / self.layers_dim)
        self.W3 = np.random.rand(self.layers_dim, y_dim) * np.sqrt(1 / layers_dim)

    def train(self, X, Y):
        # print("next sample")
        if Y.shape[0] != self.y_dim:
            Exception("Wring Y dimensions")
        if X.shape[0] != self.x_dim:
            Exception("Wring X dimensions")

        self.X = X
        self.Y = Y
        self.Y_est = np.random.rand(self.y_dim)
        # print("Training sample cost", self.cost_function())
        self.feedforward()
        self.backpropagation()

    def classify(self, X, Y):
        self.X = X
        self.Y = Y
        self.feedforward()
        # print("Cost for classify func", self.cost_function())
        return self.Y_est

    def feedforward(self):
        self.a0 = self.X
        self.z1 = np.dot(self.W1.T, self.a0) + self.b1
        self.a1 = helpers.sigmoid(self.z1)
        self.z2 = np.dot(self.W2.T, self.a1) + self.b2
        self.a2 = helpers.sigmoid(self.z2)
        self.z3 = np.dot(self.W3.T, self.a2) + self.b3
        self.a3 = helpers.softmax(self.z3)
        self.Y_est = self.a3

    def cost_function_dx(self):
        error_3 = self.a3 - self.Y
        dC_dw3 = np.outer(self.a2, error_3) + self.Lambda * self.W3

        error_2 = np.multiply(helpers.sigmoid_dx(self.z2), np.dot(self.W3, error_3))
        dC_dw2 = np.outer(self.a1, error_2) + self.Lambda * self.W2

        error_1 = np.multiply(helpers.sigmoid_dx(self.z1), np.dot(self.W2, error_2))
        dC_dw1 = np.outer(self.a0, error_1) + self.Lambda * self.W1

        return dC_dw3, error_3, dC_dw2, error_2, dC_dw1, error_1

    def accuracy(self):
        predictions = []

        for x, y in zip(self.X, self.Y):
            self.feedforward()
            pred = np.argmax(self.Y_est)
            predictions.append(pred == y)

        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed)

    def cost_function(self):
        J = -1 * np.sum(np.dot(self.Y, np.log(self.Y_est).T) + np.dot((1 - self.Y), np.log(1 - self.Y_est).T))
        L2_regularization = self.Lambda / 2 * (
                np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
        return J, L2_regularization

    def backpropagation(self):
        dC_dw3, dC_db3, dC_dw2, dC_db2, dC_dw1, dC_db1 = self.cost_function_dx()
        # print("Grad dC/dw3:", dC_dw3)
        # print("Grad dC/db3:", dC_db3)
        # print("Grad dC/dw2:", dC_dw2)
        # print("Grad dC/db2:", dC_db2)
        # print("Grad dC/dw1:", dC_dw1)
        # print("Grad dC/db1:", dC_db1)

        self.W3 -= self.learning_rate * dC_dw3
        self.b3 -= self.learning_rate * dC_db3
        self.W2 -= self.learning_rate * dC_dw2
        self.b2 -= self.learning_rate * dC_db2
        self.W1 -= self.learning_rate * dC_dw1
        self.b1 -= self.learning_rate * dC_db1
