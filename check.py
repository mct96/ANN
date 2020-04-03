import math
from random import randint, shuffle, random

def f(x):
    return 1/(1+math.exp(-x))

# def f(x):
#     return math.tanh(x)

def Df(x):
    return f(x)*(1-f(x))

# def Df(x):
#     return 1-f(x)**2

def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def xor(a, b):
    return 1 if a != b else 0

class network_t:
    def __init__(self, layout):
        self.layout = layout
        self.weight = []
        self.weight_adjustment = []
        self.sensibility = []
        self.induced_values = []
        self.output_by_layer = []
        self.error = []
        self.n_layers = len(self.layout)

    def initilizalize(self):
        self.initialize_weight()
        self.initialize_weight_adjustment()
        self.initialize_sensibility()
        self.initialize_induced_values()
        self.initialize_output_by_layer()
        self.initialize_error()

    def print(self):
        for title, data in [
            ("W", self.weight),
            ("S", self.sensibility),
            ("I", self.induced_values),
            ("O", self.output_by_layer),
            ("E", self.error),
            ("Δw", self.weight_adjustment)] :

            print(title)
            print(data)
            print("\n\n" + "-" * 80)



    def initialize_weight(self):
        for l in range(1, len(self.layout)):

            weights = []
            for i in range(self.layout[l]):
                temp = []
                for k in range(self.layout[l-1] + 1):
                    temp.append(random())
                weights.append(temp)

            self.weight.append(weights)

    def initialize_sensibility(self):
        for n in self.layout[1:]:
            self.sensibility.append([0] * n)

    def initialize_induced_values(self):
        for n in self.layout[1:]:
            self.induced_values.append([0]*n)

    def initialize_output_by_layer(self):
        for n in self.layout:
            self.output_by_layer.append([0] * n)

    def initialize_error(self):
        self.error = [float('Inf')] * self.layout[-1]

    def initialize_weight_adjustment(self):
        for l in range(1, len(self.layout)):

            weight_adjustment = []
            for i in range(self.layout[l]):
                weight_adjustment.append([0]*(self.layout[l-1] + 1))

            self.weight_adjustment.append(weight_adjustment)

    def forward(self, x):
        yi = x
        self.output_by_layer[0] = x

        for l in range(1, self.n_layers):
            for i in range(self.layout[l]):
                weights = self.weight[l-1]
                bias = weights[i][0]
                vi = dot(weights[i][1:], yi) + bias

                self.induced_values[l-1][i] = vi
                self.output_by_layer[l][i] = f(vi)

            yi = self.output_by_layer[l]


    def calcule_error(self, d):
        for i in range(self.layout[-1]):
            self.error[i] = d[i] - self.output_by_layer[-1][i]

    def calcule_sensibility(self):
        # For output layer.
        for i in range(self.layout[-1]):
            df = Df(self.induced_values[-1][i])
            self.sensibility[-1][i] = \
                self.error[i] * df

        # For hidden layer.
        for l in range(self.n_layers - 2, 0, -1):
            for i in range(self.layout[l]):
                Wk = [self.weight[l][k][i+1] for k in range(self.layout[l + 1])]
                Sk = [self.sensibility[l][k] for k in range(self.layout[l + 1])]
                self.sensibility[l-1][i] = \
                    dot(Wk, Sk) * Df(self.induced_values[l - 1][i])

    def calcule_weight_adjustment(self, η):
        for l in range(self.n_layers-1):
            for i in range(self.layout[l + 1]):
                δ = self.sensibility[l][i]

                self.weight_adjustment[l][i][0] = η * δ

                for j in range(self.layout[l]):
                    yj = self.output_by_layer[l][j]
                    self.weight_adjustment[l][i][j+1] = η * δ * yj

    def apply_adjustment(self):
        for l in range(self.n_layers-1):
            for i in range(self.layout[l+1]):
                for j in range(self.layout[l]+1): # plus bias.
                    self.weight[l][i][j] += self.weight_adjustment[l][i][j]

    def train(self, x, d, η):
        self.forward(x)
        self.calcule_error(d)
        self.calcule_sensibility()
        self.calcule_weight_adjustment(η)
        self.apply_adjustment()

    def lms(self):
        value = dot(self.error, self.error)
        return value/2

    def predict(self, x):
        yi = x

        for l in range(1, self.n_layers):
            yj = [0] * self.layout[l]
            for i in range(self.layout[l]):
                weights = self.weight[l-1]
                bias = weights[i][0]
                vi = dot(weights[i][1:], yi) + bias
                yj[i] = f(vi)

            yi = yj

        return yi

net = network_t([2, 2, 1])
net.initilizalize()

print("Training...")
for i in range(10000):
    a = randint(0, 1)
    b = randint(0, 1)
    y = xor(a, b)
    net.train([a, b], [y], .5)


print("\n\nPredicting...")
print("\tpredicted: %.3lf expected %d"%(net.predict([1, 1])[0], 0))
print("\tpredicted: %.3lf expected %d"%(net.predict([1, 0])[0], 1))
print("\tpredicted: %.3lf expected %d"%(net.predict([0, 0])[0], 0))
print("\tpredicted: %.3lf expected %d"%(net.predict([0, 1])[0], 1))
