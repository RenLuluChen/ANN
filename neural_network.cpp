// You Ren Chen
// 423006276
// CSCE 420
// Due: April 23, 2018
// neural_network.cpp

#include "neural_network.h"

Neural_Network::Neural_Network(const std::vector<int>& parameters){
    // Pushes specified number of neurons into the netwrok
    for(int i = 0; i < parameters.size(); i++){
        std::vector<Neuron> temp_vec;
        for(int j = 0; j <= parameters[i]; j++){
            int out_connections = parameters[i+1];
            // Unless output layer, create connection to next layer
            if(i == parameters.size()-1){
                out_connections = 0;
            }
            // Neuron(Output Connections,Neuron ID)
            temp_vec.push_back(Neuron(out_connections, j));
            //std::cout << "[i,j]: [" << i << "," << j << "]" << std::endl;
        }
        network.push_back(temp_vec);
        // Sets the bias neurons to output 1
        network[i].back().set_output(1.0);
        //std::cout << "Size of " << i << ": " << network[i].size() << std::endl;
    }
    //std::cin.ignore();
}
void Neural_Network::feed_forward(const std::vector<double>& input){
    // Input checking
    if(input.size() != network[0].size()-1){
        std::cout << "INPUT AND NETWORK SIZE DOES NOT MATCH!\n";
        return;
    }
    // Places the given input to the output connection of the first layer
    for(int i = 0; i < input.size(); i++){
        network[0][i].set_output(input[i]);
        //std::cout << "[" << i << "] " << network[0][i].get_output() << std::endl;
    }
    // Calls the Neuron::feed_forward for each neuron excluding bias neuron
    for(int i = 1; i < network.size(); i++){
        for(int j = 0; j < network[i].size()-1; j++){
            // Passes down the previous layer to access connection
            //std::cout << "[" << i << "," << j << "]: " << network[i].size() << std::endl;
            network[i][j].feed_forward(network[i-1]);
            //std::cin.ignore();
        }
    }
}
void Neural_Network::back_prop(const std::vector<double>& result){
    // Output checking
    if(result.size() != network.back().size()-1){
        std::cout << "Output AND NETWORK SIZE DOES NOT MATCH!\n";
        return;
    }
    // A reference to the output layer
    std::vector<Neuron>& outputs = network.back();
    // Error value reset
    error_value = 0;
    // RMS Error = sqrt((SUM((Actual-Calculated)^2))/(n-1))
    for(int i = 0; i < outputs.size()-1; i++ ){
        double diff = result[i] - outputs[i].get_output();
        error_value += pow(diff,2);
    }
    error_value /= outputs.size()-1;
    error_value = sqrt(error_value);
    // Calls Neuron::calculate_gradient for each output layer neuron
    for(int i = 0; i < outputs.size()-1; i++ ){
        outputs[i].calculate_gradient(result[i]);
    }
    // Neuron::gradient_calculation for hidden layers
    for(int i = network.size()-2; i > 0; i--){
        // Reference to current hidden layer
        std::vector<Neuron>& h_layer = network[i];
        // Reference to next layer in network
        std::vector<Neuron>& n_layer = network[i+1];
        for(int j = 0; j < h_layer.size(); j++){
            h_layer[j].calculate_hidden_gradient(n_layer);
        }
    }
    // Updates the weight of the previous layer neuron connections
    for(int i = network.size()-1; i > 0; i--){
        std::vector<Neuron>& current_layer = network[i];
        std::vector<Neuron>& previous_layer = network[i-1];
        for(int j = 0; j < current_layer.size()-1; j++){
            current_layer[j].update(previous_layer);
        }
    }
}
std::vector<double> Neural_Network::get_result() const{
    std::vector<double> result;
    for(int i = 0; i < network.back().size(); i++){
        result.push_back(network.back()[i].get_output());
    }
    return result;
}
std::vector<double> Neural_Network::activate(const std::vector<double>& input){
    feed_forward(input);
    return get_result();
}
void Neural_Network::train(const std::vector<double>& in,const std::vector<double>& out){
    feed_forward(in);
    back_prop(out);
    /*
    std::cout << "---------------------------\n";
    std::vector<double> result = get_result();
    for(int i = 0; i < result.size(); i++){
        std::cout << "i: " << result[i] << std::endl;
    }
    */
    //std::cout << "Accuracy: " << (1-get_error())*100 << "%" << std::endl;
}
double Neural_Network::get_error() const{
    return error_value;
}