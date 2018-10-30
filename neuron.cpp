// You Ren Chen
// 423006276
// CSCE 420
// Due: April 23, 2018
// neuron.cpp

#include "neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(int n, int i){
    index = i;
    for(int i = 0; i < n; i++){
        output_weights.push_back(Connection());
    }
}
double Neuron::transfer_function(double input){
    return tanh(input);
}
double Neuron::derivative_transfer_function(double input){
    return 1 - pow(input,2);
}
void Neuron::set_output(double out){
    output = out;
}
double Neuron::get_output() const{
    return output;
}
void Neuron::feed_forward(const std::vector<Neuron>& prev){
    double temp_sum = 0.0;
    // SUM(Output_i * ConnectionWeight_index)
    for(int i = 0; i < prev.size(); i++){
        temp_sum += prev[i].get_output() 
            * prev[i].output_weights[index].weight;
    }
    temp_sum /= prev.size();
    //std::cout << "Sum: " << temp_sum << std::endl;
    output = Neuron::transfer_function(temp_sum);
    //std::cout << "Output: " << output << std::endl;
    //std::cin.ignore();
}
void Neuron::calculate_gradient(const double& input){
    gradient = (input - output) * Neuron::derivative_transfer_function(output);
}
void Neuron::calculate_hidden_gradient(const std::vector<Neuron>& layer){
    double temp_sum = 0;
    for(int i = 0; i < layer.size()-1; i++){
        temp_sum += output_weights[i].weight * layer[i].gradient;
    }
    gradient = temp_sum * Neuron::derivative_transfer_function(output);
}
void Neuron::update(std::vector<Neuron>& layer){
    // Layer refers to the passed down "previous layer"
    for(int i = 0; i < layer.size(); i++){
        Neuron& current = layer[i];
        double old_weight_change = current.output_weights[index].weight_change;
        double new_weight_change = eta * current.get_output() * gradient 
                + alpha * old_weight_change;
        // Memorizes previous weight change
        current.output_weights[index].weight_change = new_weight_change;
        // Updates the new weight
        current.output_weights[index].weight += new_weight_change;
    }
}