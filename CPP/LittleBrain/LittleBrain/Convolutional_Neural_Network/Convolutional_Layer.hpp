//
//  Convolutional_Layer.hpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#ifndef Convolutional_Layer_hpp
#define Convolutional_Layer_hpp

#include "Tensor.hpp"

class Convolutional_Layer{
    public:
        //---Constructors---//
        Convolutional_Layer(void);
        Convolutional_Layer(const unsigned int numStartNodes, const unsigned int numEndNodes);
        //---Copy Constructors---//
        Convolutional_Layer(const Convolutional_Layer &obj);
        //---Destructors---//
        ~Convolutional_Layer(void);
        
        //---Get---//
        Tensor get_inputs(void);
        Tensor get_weights(void);
        Tensor get_biases(void);

        //---Set---//
        void set_inputs(Tensor data, bool doTranspose);
        void set_weights_delta(Tensor &obj);
    
        //---Operations---//
        Tensor solve_outputs(void);
        void round_to(float val);
        void randomize(int lowerBound, int upperBound, int decimal_precision = 1000);
    
        //---Data--//
        Tensor inputs;
        Tensor inputsT;
    
        Tensor weights;
        Tensor weightsT;
        Tensor biases;
    
        Tensor outputs;

        Tensor outputError;
        Tensor gradient;
        Tensor weightsDelta;
        Tensor inputError;
        //sig(W*in + b)=out
};


#endif /* Convolutional_Layer_hpp */
