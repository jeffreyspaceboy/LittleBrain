//
//  main.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/24/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

//#include <stdio.h>
#include <iostream>
//#include <sstream>
#include "Neural_Network.hpp"


std::vector<float> inputs = {
    255,  0,  0  ,  252, 43, 43  ,  209, 47, 29  ,  207, 21,  0  ,  227, 37, 16,
    107,194, 37  ,  134,242, 46  ,    6,212,  6  ,   20,227, 65  ,    0,255,  0,
      0,  0,255  ,   28,126,237  ,    8, 27,196  ,   53, 79,222  ,   23, 70,209,
    255,  0,255  ,  235, 42,196  ,  237,111,206  ,  194, 25,138  ,  245, 34,175,
    255,221,  0  ,  224,197, 20  ,  239,255, 10  ,  232,232, 19  ,  245,245, 44,
      0,255,255  ,   32,199,185  ,   58,208,235  ,    7,224,159  ,   76,245,194
}; //Input trainging data.

std::vector<float> targets = {
    1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0  ,  1,0,0,0,0,0,
    0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0  ,  0,1,0,0,0,0,
    0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0  ,  0,0,1,0,0,0,
    0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0  ,  0,0,0,1,0,0,
    0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0  ,  0,0,0,0,1,0,
    0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1  ,  0,0,0,0,0,1
}; //Correct output values for the input training data.
//
std::vector<unsigned int> numNodes = {3,8,6};
//
std::vector<float> checkInputs = {50, 120, 255};

//Red, Green, Blue, Magenta, Yellow, Cyan
int main( int argc, char** argv){
    Neural_Network myToyNN(numNodes);
    myToyNN.set_name("myToyNN");
    myToyNN.train(Tensor(inputs, Shape(3,1,30)),Tensor(targets, Shape(6,1,30)));
    myToyNN.predict(Tensor(checkInputs, Shape(3,1,1)));
    
//    NeuralNetwork myToyNN(numNodes,"networkData/colorDetector.txt");
//    myToyNN.train(inputs, targets);
//    myToyNN.predict(checkInputs);
    
    
//    std::vector<float> A = {
//        1,1,1,
//        2,2,2,
//
//        3,3,3,
//        4,4,4
//    };
//    std::vector<float> B = {
//        4,4,4,
//        3,3,3,
//
//        2,2,2,
//        1,1,1
//    };
//
//    Tensor testTensorA(A,Shape(3,2,2));
//    Tensor testTensorB(B,Shape(3,2,2));
//
//    testTensorA.print();
//    printf("\n");
//    testTensorB.print();
//    printf("\n");
//    Tensor testTensorC = ~testTensorA;
//    testTensorC.print();
    
    printf("Done\n");
    return 0;
}
