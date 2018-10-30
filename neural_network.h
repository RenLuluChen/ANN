// You Ren Chen
// 423006276
// CSCE 420
// Due: April 23, 2018
// neural_network.h

#include <vector>
#include <iostream>
#include "neuron.h"

//----------------------------------------------------
// References Used:

// Title: Neural Networks and Deep Learning Chapter 1
// Author: Nielsen, and Michael A
// Link: neuralnetworksanddeeplearning.com/chap1.html

// Title: Neural Networks and Deep Learning Chapter 2
// Author: Nielsen, and Michael A
// Link: neuralnetworksanddeeplearning.com/chap2.html

// Title: Neural Net Tutorial
// Author: Vinh Nguyen
// Link: youtube.com/watch?v=KkwX7FkLfug

//----------------------------------------------------

class Neural_Network{
public:
    // Constructor
    Neural_Network(const std::vector<int>&);
    // Destructor
    ~Neural_Network(){}
    // Takes input vector, runs it through the network
    void feed_forward(const std::vector<double>&);
    // Takes result vector and adjust neuron weight
    void back_prop(const std::vector<double>&);
    // Returns a vector of results from netwrok
    std::vector<double> get_result() const;
    // Returns the result for a given input
    std::vector<double> activate(const std::vector<double>&);
    // Accepts input and correct output, trains network
    void train(const std::vector<double>&,const std::vector<double>&);
    // Returns the error value
    double get_error() const;
private:
    // Stores the actual neurons in 2D matrix
    std::vector<std::vector<Neuron>> network;
    // Netwrok result and solution error value
    double error_value;
};
