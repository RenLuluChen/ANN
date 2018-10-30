// You Ren Chen
// 423006276
// CSCE 420
// Due: April 23, 2018
// neuron.h

#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>

struct Connection{
    // Constructor in which weight is random [0..1]
    Connection(){
        weight = double(rand()) / double(RAND_MAX);
    }
    double weight;
    double weight_change;
};

class Neuron{
public:
    // Constructor
    Neuron(int, int);
    // Destructor
    ~Neuron(){};
    // Returns tanh(x) [-1..1]
    static double transfer_function(double);
    // Approximates by returning 1-x^2
    static double derivative_transfer_function(double);
    // Sets output of neuron
    void set_output(double);
    // Gets output of neuron
    double get_output() const;
    // Obtains connection weight and neuron output, then updates output
    void feed_forward(const std::vector<Neuron>&);
    // Returns (Actual-Calculated) * transfer_function(neuron output)
    void calculate_gradient(const double&);
    // Adjusts neuron gradient based on next layer gradient and output
    void calculate_hidden_gradient(const std::vector<Neuron>&);
    // Changes the weight of the neuron connections
    void update(std::vector<Neuron>&);
private:
    // Layer index of the neuron
    int index;
    // Output of the neuron
    double output;
    // Holds current neuron connection to next layer
    std::vector<Connection> output_weights;
    // Helps neuron reach target without overshooting
    double gradient;
    // [0..1]
    static double eta;
    // [0..n]
    static double alpha;
};