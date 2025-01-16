import math
from AbstractActivationFunction import AbstractActivationFunction

class SigmoidActivation(AbstractActivationFunction):
    def __init__(self): #Empty constructor because activation function doesn't need anything
        pass
    #1/(1+e^-x)
    def Output(self,x):
        #Output is y
        return (1.0)/(1.0+math.exp(-x))
    #The derivative is equal to f(x)*(1-f(x))
    def OutputPrime(self, x):
        funcx=self.Output(x)
        return funcx*(1-funcx)