from Neuron import Neuron
from SigmoidActivation import SigmoidActivation
from Layer import Layer
from Network import Network
import numpy
import copy
from BackpropagationMethods import BackpropagationMethods

"""
sigmoid = SigmoidActivation()
neuron0 = Neuron(2,sigmoid)
neuron1 = Neuron(2, sigmoid)
neuron2 = Neuron(2, sigmoid)
neuron0.Weights = numpy.array([0.2, 0.3])
neuron1.Weights = numpy.array([0.1, 0.25])
neuron2.Weights = numpy.array([0.5, 0.6])
input= numpy.zeros(shape=(2,1))
input[0,0]=0.3
input[1,0]=0.5
output0 = neuron0.Output(input)
output1 = neuron1.Output(input)
output2 = neuron2.Output(input)
print(output0)
print(output1)
print(output2)
"""

#There are two layers, layer0 with 3 neurons of 2 inputs per neuron
#Layer 1 with 2 neurons of 3 inputs per neuron
"""
sigmoid = SigmoidActivation()
layer0= Layer(2,3,sigmoid)
layer1 = Layer(3,2,sigmoid)
layer0weights= numpy.array([[0.2,0.3],[0.1,0.25],[0.5,0.6]])
layer1weights=numpy.array([[0.8,0.1,0.7],[0.2,0.1,0.3]])
layer0.setWMatrix(layer0weights)
layer1.setWMatrix(layer1weights)
input= numpy.zeros(shape=(2,1))
input[0,0]=0.3
input[1,0]=0.5

outputLayer0 = layer0.Output(input)
outputLayer1= layer1.Output(outputLayer0)
print(outputLayer0)
print(outputLayer1)
"""

"""
numInputs=2
numNeuronPerLayer = [3,2,1]
actFunctions=[SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1 = Network(numInputs,numNeuronPerLayer, actFunctions)
input= numpy.zeros(shape=(2,1))
input[0,0]=0.3
input[1,0]=0.5

# Set the weights for the first layer
n1.Layers[0].NeuronArray[0].Weights[0] = 0.2
n1.Layers[0].NeuronArray[0].Weights[1] = 0.3
n1.Layers[0].NeuronArray[1].Weights[0] = 0.1
n1.Layers[0].NeuronArray[1].Weights[1] = 0.25
n1.Layers[0].NeuronArray[2].Weights[0] = 0.5
n1.Layers[0].NeuronArray[2].Weights[1] = 0.6

# Set the weights for the second layer
n1.Layers[1].NeuronArray[0].Weights[0] = 0.8
n1.Layers[1].NeuronArray[0].Weights[1] = 0.1
n1.Layers[1].NeuronArray[0].Weights[2] = 0.7
n1.Layers[1].NeuronArray[1].Weights[0] = 0.2
n1.Layers[1].NeuronArray[1].Weights[1] = 0.1
n1.Layers[1].NeuronArray[1].Weights[2] = 0.3

# Set the weights for the output layer
n1.Layers[2].NeuronArray[0].Weights[0] = 0.5
n1.Layers[2].NeuronArray[0].Weights[1] = 0.3



network_output = n1.Output(input)
print(network_output)
"""
"""
input=numpy.zeros(shape=(2,1)) #Allocate memory for the input
input[0,0]=0.3 #Store the value of x0
input[1,0]=0.5 #Store the value of x1
learningRate=0.5
output=0.1
numNeuronsPerLayer=[3, 2, 1]
actFunctions=[SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1=Network(2, numNeuronsPerLayer, actFunctions)
#Set dummy weights for testing
#first layer
n1.Layers[0].NeuronArray[0].Weights[0]=0.2
n1.Layers[0].NeuronArray[0].Weights[1]=0.3
n1.Layers[0].NeuronArray[1].Weights[0]=0.1
n1.Layers[0].NeuronArray[1].Weights[1]=0.25
n1.Layers[0].NeuronArray[2].Weights[0]=0.5
n1.Layers[0].NeuronArray[2].Weights[1]=0.6
#second layer
n1.Layers[1].NeuronArray[0].Weights[0]=0.8
n1.Layers[1].NeuronArray[0].Weights[1]=0.1
n1.Layers[1].NeuronArray[0].Weights[2]=0.7
n1.Layers[1].NeuronArray[1].Weights[0]=0.2
n1.Layers[1].NeuronArray[1].Weights[1]=0.1
n1.Layers[1].NeuronArray[1].Weights[2]=0.3
#Output layer
n1.Layers[2].NeuronArray[0].Weights[0]=0.5
n1.Layers[2].NeuronArray[0].Weights[1]=0.3

target_output = numpy.array([[1.0]])
beforeOutput = n1.Output(input)
BackpropagationMethods.GradientDescent(n1, input, output, learningRate)

# Print the updated weights for inspection
for i, layer in enumerate(n1.Layers):
    print(f"Layer {i} weights:")
    for neuron in layer.NeuronArray:
        print(neuron.Weights)

new_output = n1.Output(input)
print(f'This is the old output: {beforeOutput}')
print(f'This is the new output: {new_output}')
"""

"""
#First training sample
inputS1=numpy.zeros(shape=(2,1)) #Allocate memory for input
inputS1[0,0] = 0.3 #Store value of x0, x1 for first training sample
inputS1[1,0] = 0.5
outputS1 = 0.1 #Target output for sample 1
#Second sample
inputS2=numpy.zeros(shape=(2,1))
inputS1[0,0] = 0.5 #Store value of x0, x1 for second training sample
inputS1[1,0] = 0.9
outputS2 = 0.2 #Target output for sample 2
input =[] #Store input and output for each sample in array
input.append(inputS1)
input.append(inputS2)
output=[] #Array to hold all training sample target output data
output.append(outputS1)
output.append(outputS2)
#Network parameters
learningRate = 0.5
numNeuronsPerLayer= [3,2,1]
actFunctions=[SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1= Network(2,numNeuronsPerLayer,actFunctions)
#Set the dummy weights
#first layer 
n1.Layers[0].NeuronArray[0].Weights[0]=0.2 
n1.Layers[0].NeuronArray[0].Weights[1]=0.3 
n1.Layers[0].NeuronArray[0].Bias=0.0 
n1.Layers[0].NeuronArray[1].Weights[0]=0.1 
n1.Layers[0].NeuronArray[1].Weights[1]=0.25 
n1.Layers[0].NeuronArray[1].Bias=0.0 
n1.Layers[0].NeuronArray[2].Weights[0]=0.5 
n1.Layers[0].NeuronArray[2].Weights[1]=0.6 
n1.Layers[0].NeuronArray[2].Bias=0.0 
#second layer 
n1.Layers[1].NeuronArray[0].Weights[0]=0.8 
n1.Layers[1].NeuronArray[0].Weights[1]=0.1 
n1.Layers[1].NeuronArray[0].Weights[2]=0.7 
n1.Layers[1].NeuronArray[0].Bias=0.0 
n1.Layers[1].NeuronArray[1].Weights[0]=0.2 
n1.Layers[1].NeuronArray[1].Weights[1]=0.1 
n1.Layers[1].NeuronArray[1].Weights[2]=0.3 
n1.Layers[1].NeuronArray[1].Bias=0.0 
#Output layer 
n1.Layers[2].NeuronArray[0].Weights[0]=0.5 
n1.Layers[2].NeuronArray[0].Weights[1]=0.3 
n1.Layers[2].NeuronArray[0].Bias=0.0
BackpropagationMethods.GradientDescent(n1, inputS1, outputS1, learningRate)
output1=n1.Output(inputS1)
print(output1)
"""
"""
sigmoid = SigmoidActivation()
n1 = Network(2, [3, 2, 1], [sigmoid, sigmoid, sigmoid])

# Set initial weights and biases
n1.Layers[0].NeuronArray[0].Weights = numpy.array([0.2, 0.3])
n1.Layers[0].NeuronArray[1].Weights = numpy.array([0.1, 0.25])
n1.Layers[0].NeuronArray[2].Weights = numpy.array([0.5, 0.6])
n1.Layers[1].NeuronArray[0].Weights = numpy.array([0.8, 0.1, 0.7])
n1.Layers[1].NeuronArray[1].Weights = numpy.array([0.2, 0.1, 0.3])
n1.Layers[2].NeuronArray[0].Weights = numpy.array([0.5, 0.3])

# Set biases
for layer in n1.Layers:
    for neuron in layer.NeuronArray:
        neuron.Bias = 0.0

# Define input and target output
inputS1 = numpy.array([[0.3], [0.5]])
outputS1 = numpy.array([[0.1]])

# Perform one iteration of backpropagation
learningRate = 0.5
BackpropagationMethods.GradientDescent(n1, inputS1, outputS1, learningRate)

# Get the new output after training
new_output = n1.Output(inputS1)
print("New output after one iteration of backpropagation:", new_output)
"""
"""
def print_weights_and_biases(net):
    for l, layer in enumerate(net.Layers):
        print(f"Layer {l + 1} weights:")
        for neuron in layer.NeuronArray:
            print(neuron.Weights)
        print(f"Layer {l + 1} biases:")
        for neuron in layer.NeuronArray:
            print(neuron.Bias)

# Define network and initial weights and biases
sigmoid = SigmoidActivation()
n1 = Network(2, [3, 2, 1], [sigmoid, sigmoid, sigmoid])

# Print initial weights and biases
print("Initial weights and biases:")
print_weights_and_biases(n1)

# Randomize weights and biases
n1.RandomizeAll()

# Print randomized weights and biases
print("\nRandomized weights and biases:")
print_weights_and_biases(n1)
"""

inputS1 = numpy.zeros(shape=(2,1))#Allocate memory
inputS1[0,0]=0.3
inputS1[1,0]=0.5
outputS1 = 0.1
#Second sample
inputS2=numpy.zeros(shape=(2,1))
inputS2[0,0]=0.5
inputS2[1,0]=0.9
outputS2=0.2
input=[]
input.append(inputS1)
input.append(inputS2)
output=[]
output.append(outputS1)
output.append(outputS2)
learningRate=0.5
numNeuronPerLayer=[3,2,1]
actFunctions=[SigmoidActivation(),SigmoidActivation(),SigmoidActivation()]
n1=Network(2,numNeuronPerLayer,actFunctions)
#first layer 
n1.Layers[0].NeuronArray[0].Weights[0]=0.2 
n1.Layers[0].NeuronArray[0].Weights[1]=0.3 
n1.Layers[0].NeuronArray[0].Bias=0.0 
n1.Layers[0].NeuronArray[1].Weights[0]=0.1 
n1.Layers[0].NeuronArray[1].Weights[1]=0.25 
n1.Layers[0].NeuronArray[1].Bias=0.0 
n1.Layers[0].NeuronArray[2].Weights[0]=0.5 
n1.Layers[0].NeuronArray[2].Weights[1]=0.6 
n1.Layers[0].NeuronArray[2].Bias=0.0 
#second layer 
n1.Layers[1].NeuronArray[0].Weights[0]=0.8
n1.Layers[1].NeuronArray[0].Weights[1]=0.1 
n1.Layers[1].NeuronArray[0].Weights[2]=0.7 
n1.Layers[1].NeuronArray[0].Bias=0.0 
n1.Layers[1].NeuronArray[1].Weights[0]=0.2 
n1.Layers[1].NeuronArray[1].Weights[1]=0.1 
n1.Layers[1].NeuronArray[1].Weights[2]=0.3 
n1.Layers[1].NeuronArray[1].Bias=0.0 
#Output layer 
n1.Layers[2].NeuronArray[0].Weights[0]=0.5 
n1.Layers[2].NeuronArray[0].Weights[1]=0.3 
n1.Layers[2].NeuronArray[0].Bias=0.0

BackpropagationMethods.StochasticGradientDescent(n1,input,output, learningRate,1)
output2=n1.Output(inputS2)
print("wtf")
print(outputS2)
print("wht")