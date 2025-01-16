import numpy
import copy
from SigmoidActivation import SigmoidActivation

class Neuron():
    def __init__(self, numInputs, actFuctions):
        self.Weights=numpy.empty(numInputs, dtype=float) #Defines the weight for neuron
        self.WeightNum=len(self.Weights) #Variable to check the number of connections
        self.Bias =0.0
        self.ActivationFunction=copy.deepcopy(actFuctions)
        self.A=0.0 #Current output
        self.APrime=0.0 #Derivative of activiation function

    def Output(self,input):
        z= 0.0
        self.Aprime=0.0
        for i in range(0,self.WeightNum):
            z=z+input[i]*self.Weights[i]
        z=z+self.Bias
        self.A=self.ActivationFunction.Output(z)#apply the activation function
        self.APrime= self.ActivationFunction.OutputPrime(z)
        return self.A
    def RandomizeWeights(self, fanout,randGenerator):
        newWeights=[]
        for i in range(self.WeightNum):
            randomNum = randGenerator.uniform(0,1)
            newWeights.append(randomNum)
        self.Weights = numpy.array(newWeights)
    def RandomizeBias(self,randGenerator):
        self.Bias=randGenerator.uniform(0,1)
    
