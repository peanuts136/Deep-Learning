import numpy
from Network import Network

#Methods to train network via backpropagation
class BackpropagationMethods():
    #Takes in a network object as input, a target value, a learning rate, and input for the network
    def GradientDescent(net,input,y,learningRate):
        output = net.Output(input)
        deltas = [None]*net.LayerNum #Create a list to hold deltas for each layer
        #Given by error=()
        deltas[-1] = (output-y) * net.Layers[-1].OutputPrime() #Calculate the last/output layer

        for i in range(net.LayerNum - 2, -1, -1):
            layer = net.Layers[i]
            next_layer = net.Layers[i + 1]
            deltas[i] = numpy.zeros((layer.numNeurons, 1))
            for j in range(layer.numNeurons):
                delta_sum = sum(next_layer.NeuronArray[k].Weights[j] * deltas[i + 1][k]
                                for k in range(next_layer.numNeurons))
                deltas[i][j] = delta_sum * layer.OutputPrime()[j]

        # Update weights
        for i in range(net.LayerNum):
            layer = net.Layers[i]
            #Use the given input for first layer, else use the output of the previous layer
            input_to_use = input if i == 0 else net.Layers[i - 1].LayerOutput
            for j in range(layer.numNeurons):
                for k in range(layer.inputsPerNeuron):
                    layer.NeuronArray[j].Weights[k] -= learningRate * deltas[i][j] * input_to_use[k]
                #Update bias
                layer.NeuronArray[j].Bias -= learningRate * deltas[i][j]
    def ComputeAllDeltas(net,input,y):
        #Memory for delta solution
        delta=[]
        #Do forward pass
        a=net.Output(input)
        #Compute delta
        deltaIndex =0
        for i in reversed(range(net.LayerNum)):
            #print(i)
            #Output layer is handled differently than the rest
            if i==(net.LayerNum-1):
                aPrime=net.Layers[net.LayerNum-1].OutputPrime()
                delta.append(-(y-a)*aPrime)
                #Done update the delta index the first time
            else:
                #Get W matrix from the i+1 layer/1 layer ahead
                currentWeights=net.Layers[i+1].GetWMatrix()
                aPrime = net.Layers[i].OutputPrime() #get f'(z) from current ith layer
                currentDelta = numpy.multiply(numpy.dot(currentWeights.T, delta[deltaIndex]), aPrime)
                delta.append(currentDelta)
                #delta[L]= weight[L]^T*delta[L+1] *neuronOutputPrime[L]
                delta.append(currentDelta)
                deltaIndex +=1
        return delta[::-1]
    def StochasticGradientDescent(net,inputs,targets,learningRate, numIterations):
        trainingSampleNum=len(inputs) #Determine how many samples are in the training set
        for currentIteration in range(0, numIterations):
            for currentSample in range(0,trainingSampleNum):
                currentInput=inputs[currentSample] #Get the current training input into the network
                currentTarget = targets[currentSample] #Get the ideal target output of the network
                output= net.Output(currentInput)
                #TO DO: Compute the deltas for current sample
                deltas = BackpropagationMethods.ComputeAllDeltas(net,currentInput,currentTarget)
                #Compute the partial derivative of error with respect to the weights based on the deltas
                for i in range(net.LayerNum):
                    layer= net.Layers[i]
                    #Use the given input for the first layer else use the output of the previous layer 
                    if i==0:
                        inputToUse = currentInput
                    else:
                        inputToUse = net.Layers[i-1].LayerOutput
                    for j in range(layer.numNeurons):
                        for k in range(layer.inputsPerNeuron):
                            weightGradient = deltas[i][j] * inputToUse[k]
                            layer.NeuronArray[j].Weights[k] -= learningRate * weightGradient
                        biasGradient = deltas[i][j]
                        layer.NeuronArray[j].Bias -= learningRate * biasGradient
                #Compute the partial derivative of error with respect to the bias based on deltas
                #Update the weights ofthe network according to partial derivatives and learning rate
                #Update the biases of the network according to the deltas and learning rate
    def BatchGradientDescent(net,inputs,targets,learningRate,numIterations):
        trainingSampleNum=len(inputs) #Determine amount of samples
        for currentIteration in range(0,numIterations): #Train for a fixed num of samples
            holderJW =BackpropagationMethods.GenerateEmptyJWHolder(net)
            holderJB =BackpropagationMethods.GenerateEmptyJBHolder(net)
            for currentSample in range(0,trainingSampleNum): #Train for each sample
                currentInput=inputs[currentSample] #Get the current training input to the network
                currentTarget=targets[currentSample] #Get the ideal target of the network for sample
                output=net.Output(currentInput)
                #To do: Compute weight partial derivative for the current sample and store it using SumHolders method
                #To do:Compute bias partial derivative and store using sumholders method
                BackpropagationMethods.ApplyAverageToHolders(trainingSampleNum, holderJW)
                BackpropagationMethods.ApplyAverageToHolders(trainingSampleNum, holderJB)
                #To Do: Apply holderJW to update the weights of the network
                #To do: apply holderJB to update the biases of the network
    #Generates an empty holder with the correct size numpy matrices for weights for batch gradient descent to use
    def GenerateEmptyJWHolder(net): 
        EmptyJWHolder=[]
        for i in reversed(range(net.LayerNum)):
            matrixJWForLayer=numpy.zeros(shape=(net.Layers[i].NumNeurons,net.Layers[i].InputsPerNeuron))
            matrixJWForLayer = numpy.matrix.transpose(matrixJWForLayer) #Transform so that it is in same form as in matrix returned by computeJW
            EmptyJWHolder.append(matrixJWForLayer)
        return EmptyJWHolder
    
    #Generates an empty holder with the correct size numpy matrices for biases for batch gradient descent to use
    def GenerateEmpttyJBHolder(net):
        EmptyJBHolder=[]
        for i in reversed(range(net.LayerNum)):
            matrixJBForLayer=numpy.zeros(shape=(net.Layers[i].NumNeuron,1))
            EmptyJBHolder.append(matrixJBForLayer)
        return EmptyJBHolder
    #Multiplies the summed deltas by a 1/m term
    def ApplyAverageToHolders(trainingSampleNum,holderJ):
        averagingTerm=1/trainingSampleNum
        for i in range(0,len(holderJ)):
            holderJ[i]=holderJ[i]*averagingTerm
    #Adds the matrix of JW or JB together for batch gradient descent
    def SumHolder(currentJW, holderJW):
        for i in range(0,len(currentJW)):
            holderJW[i]=numpy.add(currentJW[i],holderJW[i])

    