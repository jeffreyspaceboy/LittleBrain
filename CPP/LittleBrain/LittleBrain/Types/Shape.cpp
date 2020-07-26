//
//  Shape.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#include "Shape.hpp"

//---Constructors---//
Shape::Shape(){
    size = 0;
    dim.push_back(0);
    dim.push_back(0);
    dim.push_back(0);
}

Shape::Shape(unsigned int x_size, unsigned int y_size, unsigned int z_size){
    size = x_size * y_size * z_size;
    dim.push_back(x_size);
    dim.push_back(y_size);
    dim.push_back(z_size);
}

//---Copy Constructors---//
Shape::Shape(const Shape &obj){
    size = obj.size;
    for(unsigned int i = 0; i < obj.dim.size(); i++){
        dim.push_back(obj.dim[i]);
    }
}

//---Destructors---//
Shape::~Shape(void){}
