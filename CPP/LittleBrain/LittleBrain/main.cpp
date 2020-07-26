//
//  main.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/24/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#include <iostream>

#include "Neural_Network.hpp"

std::vector<float> training_inputs = {
    255,  0,  0  ,  252, 43, 43  ,  209, 47, 29  ,  207, 21,  0  ,  227, 37, 16, //RED
    107,194, 37  ,  134,242, 46  ,    6,212,  6  ,   20,227, 65  ,    0,255,  0, //GREEN
      0,  0,255  ,   28,126,237  ,    8, 27,196  ,   53, 79,222  ,   23, 70,209, //BLUE
    255,  0,255  ,  235, 42,196  ,  237,111,206  ,  194, 25,138  ,  245, 34,175, //MAGENTA
    255,221,  0  ,  224,197, 20  ,  239,255, 10  ,  232,232, 19  ,  245,245, 44, //YELLOW
      0,255,255  ,   32,199,185  ,   58,208,235  ,    7,224,159  ,   76,245,194  //CYAN
}; //Input trainging data.

std::vector<float> training_targets = {
    1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0, //RED
    0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0, //GREEN
    0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0, //BLUE
    0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0, //MAGENTA
    0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0, //YELLOW
    0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1  //CYAN
}; //Correct output values for the input training data.

std::vector<unsigned int> neural_layers = {3,8,6};

std::vector<float> test_input = {50, 120, 255};

int main( int argc, char** argv){
    Neural_Network lilBrain(neural_layers);
    lilBrain.set_name("lilBrain");
    lilBrain.train(Tensor(training_inputs, Shape(3,1,30)),Tensor(training_targets, Shape(6,1,30)));
    lilBrain.predict(Tensor(test_input, Shape(3,1,1)));
    printf("Done\n");
    return 0;
}
