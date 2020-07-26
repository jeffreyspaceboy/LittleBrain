//
//  Convolutional_Neural_Network.hpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#ifndef Convolutional_Neural_Network_hpp
#define Convolutional_Neural_Network_hpp

#include <fstream>
#include <vector>

#include "Convolutional_Layer.hpp"

class Convolutional_Neural_Network{
    public:
        //---Constructors---//
        Convolutional_Neural_Network(void);
        Convolutional_Neural_Network(std::vector<unsigned int> neurons);
        Convolutional_Neural_Network(std::string name);
        //---Copy Constructors---//
        Convolutional_Neural_Network(const Convolutional_Neural_Network &obj);
        Convolutional_Neural_Network(const Convolutional_Neural_Network &obj, std::string new_name);
        //---Destructors---//
        ~Convolutional_Neural_Network(void);
    
        //---Set---//
        void set_name(std::string name);
        //---Get---//
        std::string get_name(void);
    
    
        //---File_Write---//
        void save_file(void);
        void save_file(std::string name);
        //---File_Read---//
        void get_file(void);
        void get_file(std::string name);
        void setup_from_file(void);
    
    
        //---Predict---//
        Tensor feed_forward(Tensor input_data);
        Tensor predict(Tensor input_data);
        //---Learn---//
        void train(Tensor input_data, Tensor target_data);
    
    private:
        //---Network_Structure---//
        std::vector<unsigned int> neurons; //Address = layer, Value = #nodes/layer
        std::vector<Convolutional_Layer> layers;
     
        //---Settings---//
        std::string name = "Convolutional_Neural_Network";
        float learning_rate = 0.2;
        unsigned int training_runs = 1000;
};

#endif /* Convolutional_Neural_Network_hpp */
