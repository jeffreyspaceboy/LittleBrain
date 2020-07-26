//
//  Neural_Layer.hpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#ifndef Neural_Layer_hpp
#define Neural_Layer_hpp

#include "Tensor.hpp"

class Neural_Layer{
    public:
        //---Constructors---//
        Neural_Layer(void);
        Neural_Layer(const unsigned int numStartNodes, const unsigned int numEndNodes);
        //---Copy Constructors---//
        Neural_Layer(const Neural_Layer &obj);
        //---Destructors---//
        ~Neural_Layer(void);
        
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

#endif /* Neural_Layer_hpp */
