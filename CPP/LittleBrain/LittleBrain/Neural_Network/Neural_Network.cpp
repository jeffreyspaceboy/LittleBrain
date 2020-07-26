//
//  Neural_Network.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#include "Neural_Network.hpp"

//---Constructors---//
Neural_Network::Neural_Network(void){}

Neural_Network::Neural_Network(std::vector<unsigned int> neurons){
    this->neurons = neurons;
    for(unsigned int i = 0; i < (this->neurons.size()-1); i++){
        this->layers.push_back(Neural_Layer(this->neurons[i], this->neurons[i+1]));
    }
}

Neural_Network::Neural_Network(std::string name){
    this->name = name;
    this->get_file();
}


//---Copy Constructors---//
Neural_Network::Neural_Network(const Neural_Network &obj){
    this->name = ("Copy_of_" + obj.name);
    this->neurons = obj.neurons;
    for(int i=0; i < (this->neurons.size()-1) ; i++){
        this->layers.push_back(Neural_Layer(this->neurons[i], this->neurons[i+1]));
    }
}

Neural_Network::Neural_Network(const Neural_Network &obj, std::string new_name){
    this->name = new_name;
    this->neurons = obj.neurons;
    for(int i=0; i < (this->neurons.size()-1) ; i++){
        this->layers.push_back(Neural_Layer(this->neurons[i], this->neurons[i+1]));
    }
}


//---Destructors---//
Neural_Network::~Neural_Network(void){
    //this->save_file(this->name);
}


//---Set---//
void Neural_Network::set_name(std::string name){ this->name = name; }

//---Get---//
std::string Neural_Network::get_name(void){ return this->name; }


//---File_Write---//
void Neural_Network::save_file(void){ this->save_file(this->name); }

void Neural_Network::save_file(std::string name){
    /* File Format
        Array Containing Neruon Structure
        ~
        Current_Layer   Select_Weights   X_Size   Y_Size   Z_Size
        Row of Weights Data
        ~
        Current_Layer    Select_Biases   X_Size   Y_Size   Z_Size
        Row of BiasesData

        ~
        ...
    */
    std::ofstream file;
    file.open(name);
    if(!file.is_open()){
        printf("ERROR: File can not be opened.\n");
        exit(1);
    }
    unsigned int i,j;
    for(i = 0; i < this->neurons.size(); i++){
        file << this->neurons[i] << " ";
    }
    file << "\n";
    for(i = 0; i < this->layers.size(); i++){
        file << "~\n" << i << " " << 0 <<" ";
        Shape weights_shape(this->layers[i].weights.get_shape());
        for(j = 0; j < sizeof(weights_shape.dim); j++){
            file << weights_shape.dim[j] << " ";
        }
        file << "\n";
        for(j = 0; j < weights_shape.size; j++){
            file << this->layers[i].weights.get_cell(j)<<" ";
        }
        file << "\n~\n" << i << " " << 1 <<" ";
        Shape biases_shape(this->layers[i].biases.get_shape());
        for(j = 0; j < sizeof(biases_shape.dim); j++){
            file << biases_shape.dim[j] << " ";
        }
        file << "\n";
        for(j = 0; j < biases_shape.size; j++){
            file << this->layers[i].biases.get_cell(j)<<" ";
        }
        file << "\n";
    }
    file.close();
}


//---File_Read---//
void Neural_Network::get_file(void){ this->get_file(this->name); }

void Neural_Network::get_file(std::string name){
//    int neuronsInFile = this->checkNumNodesInFile();
//    if(neuronsInFile == 0){
//        std::cout<<"WARNING: File data dimentions do not match this Neural Network."<<std::endl;
//        std::cout<<"Would you like to override your current network? (y:n)"<<std::endl;
//        char ans;
//        std::cin >> ans;
//        if(ans=='y' || ans=='Y'){
//            this->layers.clear();
//            this->neurons.clear();
//            this->setupNetworkFromFile();
//        }
//    }
    /* File Format
        Array Containing Neruon Structure
        ~
        Current_Layer   Select_Weights   X_Size   Y_Size   Z_Size
        Row of Weights Data
        ~
        Current_Layer    Select_Biases   X_Size   Y_Size   Z_Size
        Row of BiasesData

        ~
        ...
    */
    std::ifstream file;
    file.open(name);
    if(!file.is_open()){
        printf("ERROR: File can not be opened.\n");
        exit(1);
    }
    unsigned int matrixInfo[5];
    std::string scannedValue;
    while(scannedValue != "\n"){
        std::getline(file, scannedValue, ' ');
        neurons.push_back(std::stoi(scannedValue));
    }
    unsigned int i,j;
    while(!file.eof()){
        std::getline(file, scannedValue);
        if(scannedValue == "~"){
            for(j = 0; j < 5; j++){
                std::getline(file, scannedValue, ' ');
                matrixInfo[j] = std::stoi(scannedValue);
            }
            for(i = 0; i < (matrixInfo[2] * matrixInfo[3] * matrixInfo[4]); i++){
                std::getline(file, scannedValue, ' ');
                if(matrixInfo[1] == 0){
                    this->layers[matrixInfo[0]].weights.set_cell(std::stod(scannedValue), i);
                }else if(matrixInfo[1] == 1){
                    this->layers[matrixInfo[0]].biases.set_cell(std::stod(scannedValue), i);
                }
            }
        }
    }
}



void Neural_Network::setup_from_file(void){
    std::ifstream file;
    file.open(this->name);
    if(file.is_open()){
        int matrixInfo[4];
        std::string scannedValue;
        std::vector<unsigned int> neurons;
        while(!file.eof()){
            std::getline(file, scannedValue);
            if(scannedValue == "~"){
                for(int j=0;j<4;j++){
                    std::getline(file, scannedValue, ' ');
                    matrixInfo[j] = std::stoi(scannedValue);
                }
                if(matrixInfo[1]==0){
                    if(matrixInfo[0] == 0){
                        neurons.push_back(matrixInfo[3]);
                    }
                    neurons.push_back(matrixInfo[2]);
                }
            }
        }
        this->neurons = neurons;
        file.close();
    } else {
        std::cout<<"ERROR: Could not open file."<<std::endl;
        exit(1);
    }
    for(int i=0; i < (this->neurons.size()-1) ; i++){
        Neural_Layer newLayer(this->neurons[i], this->neurons[i+1]);
        this->layers.push_back(newLayer);
    }
}



//---Predict---//
Tensor Neural_Network::feed_forward(Tensor input_data){
    this->layers.front().set_inputs(input_data, true);
    for(unsigned int i = 1; i < this->layers.size(); i++){
        this->layers[i].set_inputs(this->layers[i-1].solve_outputs(), false);
    }
    return (this->layers[this->layers.size()-1].solve_outputs());
}

Tensor Neural_Network::predict(Tensor input_data){
    if(input_data.get_shape().dim[0] != this->neurons.front()){
        printf("ERROR: Input data does not match network input size.\n");
        exit(1);
    }
    Tensor finalOutput(this->feed_forward(input_data));
    finalOutput.print();
    return finalOutput;
}


//---Learn---//
void Neural_Network::train(Tensor input_data, Tensor target_data){
    if((input_data.get_shape().dim[0] != this->neurons.front()) || (target_data.get_shape().dim[0] != this->neurons.back())){
        std::cout<<"ERROR: Training data dimentions do not match network dimentions."<<std::endl;
        exit(1);
    }
    unsigned int x, i;
    unsigned long int k;
    for(i = 0; i < training_runs; i++){
        x = rand() % (input_data.get_shape().dim[2] - 1);
        Tensor trainingOutputs(this->feed_forward(input_data.get_matrix(x)));
        this->layers[this->layers.size()-1].outputError = (~target_data.get_matrix(x)) - trainingOutputs;
        this->layers[this->layers.size()-1].outputs = trainingOutputs;
        for(k = (this->layers.size() - 1); k > 0 ; k--){
            this->layers[k].gradient = this->layers[k].outputs;
            this->layers[k].gradient.dSigmoid();
            this->layers[k].gradient = (this->layers[k].gradient ->* this->layers[k].outputError) * this->learning_rate;
            this->layers[k].inputsT = ~(this->layers[k].inputs);
            Tensor delt(this->layers[k].gradient * this->layers[k].inputsT);
            this->layers[k].set_weights_delta(delt);
            this->layers[k].weights = this->layers[k].weights + this->layers[k].weightsDelta;
            this->layers[k].biases = this->layers[k].biases + this->layers[k].gradient;
            this->layers[k].weightsT = ~(this->layers[k].weights);
            this->layers[k].inputError = this->layers[k].weightsT * this->layers[k].outputError;
            this->layers[k-1].outputError = this->layers[k].inputError;
            this->layers[k-1].outputs = this->layers[k].inputs;
        }
    }
}



