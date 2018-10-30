// You Ren Chen
// 423006276
// CSCE 420
// Due: April 23, 2018
// main.cpp

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "neural_network.h"
#include "font5x7.h"

using namespace std;

//-------------------------------------------------------
// Functions
//-------------------------------------------------------
// Prints the letter specified by a vector of doubles
void print_letter(const vector<double>& letter){
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 5; j++){
            if(letter[i*5+j] > 0){
                cout << "\u25A0" << " ";
            }
            else{
                cout << "  ";
            }
        }
        cout << endl;
    }
}
// Returns a vector containing vectors that represents each letter
vector<vector<double>> get_test_data(){
    vector<vector<double>> data;
    for(int i = 0; i < 26; i++){
        vector<double> temp;
        for(int j = 0; j < 7; j++){
            for(int k = 0; k < 5; k++){
                int new_val = (Font5x7[i*5+k] & (0x01 << j)) >> j;
                temp.push_back((double)new_val);
            }
        }
        data.push_back(temp);
        //cout << "--------------------------\n";
        //print_letter(temp);
    }
    return data;
}
// Takes a 5 element user input of ints and convert it into something usable by the neural net
vector<double> format(const vector<int>& input){
    vector<double> data;
    for(int j = 0; j < 7; j++){
        for(int k = 0; k < 5; k++){
            int new_val = (input[k] & (0x01 << j)) >> j;
            data.push_back((double)new_val);
        }
    }
    return data;
}
// Returns a vector of vectors to be used as training solution.
vector<vector<double>> get_solution(){
    vector<vector<double>> solution;
    for(int i = 0; i < 26; i++){
        vector<double> temp;
        for(int j = 0; j < 26; j++){
            if(i == j){
                temp.push_back(1.0);
            }
            else{
                temp.push_back(0.0);
            }
            //cout << temp[j];
        }
        solution.push_back(temp);
        //cout << endl;

    }
    return solution;
}
// Outputs the data, also returns whether the neural net was correct
int output_solution(const vector<double>& solution, const vector<char>& letters, int answer){
    int selection = -1;
    double max = 0.0;
    for(int i = 0; i < solution.size()-1; i++){
        if(solution[i] > max){
            max = solution[i];
            selection = i;
        }
    }
    if(selection == -1){
        cout << "No Solution Found!\n";
        return 0;
    }
    cout << "The network believes '" << letters[selection] << "'.\n";
    cout << "The network is " << solution[selection] * 100 << "% sure.\n";
    cout << "The correct answer is '" << letters[answer] << "'.\n";
    if(selection == answer){
        cout << "<CORRECT>\n";
        return 1;
    }
    else{
        cout << "<WRONG>\n";
        return 0;
    }
}
// Simplified override of output_solution, does not return whether the user is correct
void output_solution(const vector<double>& solution, const vector<char>& letters){
    int selection = -1;
    double max = 0.0;
    for(int i = 0; i < solution.size()-1; i++){
        if(solution[i] > max){
            max = solution[i];
            selection = i;
        }
    }
    if(selection == -1){
        cout << "No Solution Found!\n";
        return;
    }
    cout << "The network believes '" << letters[selection] << "'.\n";
    cout << "The network is " << solution[selection] * 100 << "% sure.\n";
}
// Flips a specified number of 'pixels' based on parameters
vector<double> distort(vector<double> input, int pixel_num){
    int index;
    for(int i = 0; i < pixel_num; i++){
        index = rand() % input.size();
        if(input[index] == 0){
            input[index] = 1;
        }
        else{
            input[index] = 0;
        }
    }
    return input;
}
// The entire project function
void project(){
    // Acquiring training data
    vector<vector<double>> test_data = get_test_data();
    vector<vector<double>> solution = get_solution();
    vector<char> letters = {'A','B','C','D','E','F','G',
                            'H','I','J','K','L','M','N',
                            'O','P','Q','R','S','T','U',
                            'V','W','X','Y','Z'};
    // Setting up network
    cout << "-------------------------------------------\n";
    cout << "Welcome to my project Neural Network!\n";
    cout << "-------------------------------------------\n";
    cout << "\nCreating Neural Network...\n";
    vector<int> parameters = {35,35,26};
    Neural_Network n_net(parameters);
    cout << "Done!\n";
    double desired_accuracy = 0.96;
    cout << "\n>Default training accuracy is set to " << desired_accuracy*100 << "%.\n";
    cout << "Press [Enter] to start training.";
    cin.ignore();

    // Training Neural Network
    cout << "\nTraining...\n";
    double test_accuracy = 0.0;
    int iteration = 0;
    while(test_accuracy < desired_accuracy){
        test_accuracy = 0.0;
        iteration++;
        for(int i = 0; i < test_data.size(); i++){
            n_net.train(test_data[i],solution[i]);
            test_accuracy += n_net.get_error();
        }
        test_accuracy /= test_data.size();
        test_accuracy = 1-test_accuracy;
        cout << "[" << iteration << "] Average Training Accuracy: " << test_accuracy*100 << "%\n";
    }
    cout << "\nDone Training!\n\n";
    cout << "Press [Enter] to start standard testing.";
    cin.ignore();

    // Standard Test without distortion
    cout << "--------------------\n";
    cout << "Standard Test\n";
    cout << "--------------------\n";
    cout << "\nTesting...\n";
    vector<double> test_sol;
    double avg_acc = 0;
    for(int i = 0; i < 26; i++){
        cout << "-----------------------\n";
        cout << "Test Number " << i << ":\n";
        cout << "-----------------------\n";
        test_sol = n_net.activate(test_data[i]);
        print_letter(test_data[i]);
        avg_acc += output_solution(test_sol,letters,i);
    }
    avg_acc /= 26.0;
    cout << "\n--------------------------------------------\n";
    cout << "Average accuracy is: " << avg_acc*100 << "%\n";
    cout << "--------------------------------------------\n\n";
    cout << "Press [Enter] to start distortion recognition test.";
    cin.ignore();

    // Test with pixel distortion
    cout << "\nTesting for Distortion Recognition...\n";
    int pixel_num = 0;
    string temp;
    cout << "Indicate the postive number of 'pixels' to distort: ";
    cin >> temp;
    pixel_num = stoi(temp);
    cout << "\n--------------------\n";
    cout << "Distortion Test\n";
    cout << "--------------------\n";
    cout << "\nTesting...\n";
    avg_acc = 0;
    for(int i = 0; i < 26; i++){
        cout << "-----------------------\n";
        cout << "Test Number " << i << ":\n";
        cout << "-----------------------\n";
        vector<double> temp_test_data = distort(test_data[i],pixel_num);
        test_sol = n_net.activate(temp_test_data);
        print_letter(temp_test_data);
        avg_acc += output_solution(test_sol,letters,i);
    }
    avg_acc /= 26.0;
    cout << "\n--------------------------------------------\n";
    cout << "Average accuracy is: " << avg_acc*100 << "%\n";
    cout << "--------------------------------------------\n\n";

    cout << "Press [Enter] to start Max Distortion Tests.\n";
    cin.ignore();
    cin.ignore();
    
    // Test for Max Distortion.
    for(int i = 0; i < 26; i++){
        cout << "\n----------------------------\nTesting for " 
        << letters[i] << "\n----------------------------\n";
        for(int j = 0; j < 35; j++){
            vector<double> temp_test_data = distort(test_data[i],j);
            test_sol = n_net.activate(temp_test_data);
            print_letter(temp_test_data);
            if(!output_solution(test_sol,letters,i)){
                cout << "\n----------------------------\n";
                cout << "Maximum distortion for " << letters[i] << " is " << j-1 << ".\n";
                cout << "----------------------------\n";
                break;
            }
        }
    }


    cout << "\nPress [Enter] to start user input tests.\n";
    cin.ignore();

    // User input tests
    cout << "Example: 126 17 17 17 126 represents 'A'.\n";
    cout << "Be sure to press [enter] between each integer.\n";
    while(1){
        cout << "Input a negative number to quit.\n";
        cout << "Please input 5 decimal numbers that represent the letter:\n";
        vector<int> user_input;
        for(int i = 0; i < 5; i++){
            int temp_input;
            cout << '[' << i+1 << "] ";
            cin >> temp_input;
            if(temp_input < 0){
                cout << "\nQuitting...\n";
                return;
            }
            user_input.push_back(temp_input);
        }
        cout << "\n--------------------------------\n";
        cout << "User Input";
        cout << "\n--------------------------------\n";
        vector<double> formatted_user_input = format(user_input);
        print_letter(formatted_user_input);
        test_sol = n_net.activate(formatted_user_input);
        output_solution(test_sol,letters);
        cout << "\n--------------------------------\n\n";
    }
    
}

//-------------------------------------------------------
// Main
//-------------------------------------------------------
int main(){
    project();
}