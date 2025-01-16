from Layer import Layer
import random
import copy

class Network():
    def __init__(self, numInputs, numNeuronsPerLayer, actFunctionForLayer):
        self.LayerNum = len(numNeuronsPerLayer) #Number of layers
        self.Layers=[] #Variable to hold layers
        #First layer set differently
        self.Layers.append(Layer(numInputs, numNeuronsPerLayer[0], actFunctionForLayer[0]))
        #Fill in rest of layers
        for i in range(1,self.LayerNum):
            currentLayer = Layer(self.Layers[i-1].numNeurons,numNeuronsPerLayer[i], actFunctionForLayer[i])
            self.Layers.append(currentLayer)
        self.RandomizeAll() #Randomize all the weights and biases of network, based on the sigmoid act function
    def Output(self,input):
        currentOut=self.Layers[0].Output(input)
        for i in range(1, self.LayerNum):
            currentOut=self.Layers[i].Output(currentOut)
        return copy.deepcopy(currentOut)
    def RandomizeAll(self):
        for i in range(0,self.LayerNum):
            if(i==(self.LayerNum-1)): #Last layer is different
                self.Layers[i].RandomizeLayerWeights(1,random)
                self.Layers[i].RandomizeLayerBias(random)#each neuron in output is directly used so fanout=1
            else:
                self.Layers[i].RandomizeLayerWeights(self.Layers[i+1].numNeurons, random)
                self.Layers[i].RandomizeLayerBias(random)
                