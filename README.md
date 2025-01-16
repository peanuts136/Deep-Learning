# Deep-Learning
A simple neural network utilizing the classic deep learning model showcasing how neurons, layers, and networks interact to perform forward and backward propagation. Also, this project utilizes activations functions such as a sigmoid which all is combined in class structures for abstraction

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Files](#Files)
  - [1. Abstract Activation Function](#1-convex-hull-problem)
  - [2. Sigmoid Activation](#3-loaded-vs-fair-die-problem)
  - [2. Neuron](#3-loaded-vs-fair-die-problem)
  - [2. Layer](#3-loaded-vs-fair-die-problem)
  - [2. Network](#3-loaded-vs-fair-die-problem)
  - [2. Backpropagation Methods](#3-loaded-vs-fair-die-problem)

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
### abstractActivation.py
- Purpose: to define an abstract function for any activation function to be chosen
- Usage: to be inherited activation function, in this case, the sigmoid function. 
### sigmoidActivation.py

### neuron.py

### layer.py

### network.py

### backpropagationMethods.py 




