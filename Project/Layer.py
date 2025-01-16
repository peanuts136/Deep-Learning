from Neuron import Neuron
import numpy
import copy

class Layer():
    def __init__(self,inputsPerNeuron,numNeurons,actFunction):
        self.numNeurons=numNeurons #Number of neurons in the layer
        self.inputsPerNeuron = inputsPerNeuron #Number of inputs each neuron takes in
        self.LayerOutput=numpy.zeros(shape=(numNeurons,1)) #Creates a 1D array with the length of numNeurons
        self.LayerOutputPrime = numpy.zeros(shape=(numNeurons,1)) #Will store the derivative of the activation functions/neuron
        self.WMatrix = numpy.zeros(shape=(numNeurons,inputsPerNeuron))
        self.BMatrix = numpy.zeros(shape=(numNeurons,1))
        #First index is the neuron, second is the weight for neurons
        self.NeuronArray=[]
        for i in range(self.numNeurons):
            self.NeuronArray.append(Neuron(inputsPerNeuron,actFunction))
    def Output(self,inputArray):
            if inputArray is None:
                for i in range(0,self.numNeurons):
                    self.LayerOutput[i]=self.NeuronArray[i].A
            else:
                for i in range(0,self.numNeurons):
                    self.LayerOutput[i]=self.NeuronArray[i].Output(inputArray)
            return self.LayerOutput
    def OutputPrime(self):
        for i in range(0,self.numNeurons):
            self.LayerOutputPrime[i]=self.NeuronArray[i].APrime
        return self.LayerOutputPrime
    def GetWMatrix(self):
        for i in range(0,self.numNeurons):
            for j in range(0,self.inputsPerNeuron):
                self.WMatrix[i,j]=self.NeuronArray[i].Weights[j]
        return self.WMatrix
    def setWMatrix(self,updatedWMatrix):
        self.WMatrix=copy.deepcopy(updatedWMatrix)
        for i in range(0,self.numNeurons):
            updatedWeightsForNeuron=updatedWMatrix[i,:]
            self.NeuronArray[i].Weights=updatedWeightsForNeuron
    def getBMatrix(self):
        for i in range(0,self.numNeurons):
            self.BMatrix[i,0]=self.NeuronArray[i].Bias
        return self.BMatrix
    def setBMatrix(self,updatedBMatrix):
        self.BMatrix=copy.deepcopy(updatedBMatrix)
        for i in range(0, self.numNeurons):
            self.NeuronArray[i].Bias=updatedBMatrix[i,0]
    def RandomizeLayerWeights(self,fanOut,randGenerator):
        for i in range(0,self.numNeurons):
            self.NeuronArray[i].RandomizeWeights(fanOut,randGenerator)
        self.GetWMatrix()
    def RandomizeLayerBias(self,randGenerator):
        for i in range(0,self.numNeurons):
            self.NeuronArray[i].RandomizeBias(randGenerator)
        self.getBMatrix()