#The abstract class that all activation functions should inherit from
from abc import ABCMeta, abstractmethod #ABC=abstract base class

class AbstractActivationFunction(metaclass=ABCMeta):
    def Output(self,x): #Output of activation function from input x
        raise NotImplementedError("The activation function you are using isn't implemented for output method")
    def OutputPrime(self,x):#Output the derivative of the activation function from input x
        raise NotImplementedError("The activation function you are using isn't implemented for outputPrime method")