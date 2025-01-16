# Deep-Learning
A simple neural network utilizing the classic deep learning model showcasing how neurons, layers, and networks interact to perform forward and backward propagation. Also, this project utilizes activations functions such as a sigmoid which all is combined in class structures for abstraction

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Files](#Files)
  - [1. Abstract Activation Function](#1-abstract-activation-function)
  - [2. Sigmoid Activation](#2-sigmoid-activation)
  - [3. Neuron](#3-neuron)
  - [4. Layer](#4-layer)
  - [5. Network](#5-network)
  - [6. Backpropagation Methods](#6backpropagation-methods)

---

## Overview
This project is meant to demonstrate a basic deep learning model from scratch, utilizing the core components of a neural network
1. *Neurons* process the many weighted inputs and outputs
2. *Layers* organizes many neurons into a large structure
3. *Networks* organizes many layers together
4. *Activation Functions* that utilizes the sigmoid function specifically
5. *Backpropagation methods* to update the weights based on the errors

---

## Project Structure
- abstractActivation.py : base abstract class to use an activation function
- sigmoidActivation.py : implements the sigmoid activation function
- neuron.py : implements the structure of individual neuron
- layer.py : defines a layer which can hold many neurons
- network.py : defines a network which can combine many layers into a complete neural network
- backpropagationMethods.py : methods to train the network using backpropagation

---

## Files
### 1. Abstract Activation Function
- Purpose: to define an abstract function for any activation function to be chosen
- Key Methods:
  -Output(x): abstract method to be implemented to compute the activation function
  -OutputPrime(x): abstract method to be implemented to compute the derivative of the activation funciton
- Usage: to be inherited activation function, in this case, the sigmoid function. 
### 2. Sigmoid Activation
- Definition: a sigmoid function is a mathematical function in the form σ(x)=1/(1+e^(−x)​). This creates an "S-shaped" curve which takes an input of x and outputs a value between 0 and 1.
- Purpose: to implement the activation function as well as it's derivative function
- Key Methods:
  -Output(x): returns the sigmoid of x
  -OutputPrime(x): return the derivative of the sigmoid of x
- Usage: pass into the neuron to define how the neuron should transform input and compute its derivative
### 3. Neuron
- Purpose: defines a single neuron which includes the follow elements:
  - Weights: an array of weights for each input connection
  - Bias: a scalar bias added to the weighted sum
  - Activation Function: in this case the sigmoid function
  - Current Output(A): the neuron's activation after the forward pass
  - Current Derivative(APrime): the derivative of the activation funcion at the current input
- Key Methods:
  - Output(input): computes z = sum(xi * wi) + Bias, then applying the activation function
  - RandomizeWeights(fanout, randGenerator): generates random initialization weights
  - RandomizeBias(randGenerator): initializies the bias randomly
- Usage: to be called by the layer class
### 4. Layer
- Purpose:
- Key Methods:
- Usage: 
### 5. Network
- Purpose:
- Key Methods:
- Usage: 
### 6. Backpropagation Methods




