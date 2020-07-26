//
//  Cell.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#include "Cell.hpp"

//---Constructors---//
Cell::Cell(unsigned int x, unsigned int y, unsigned int z){
    p.push_back(x);
    p.push_back(y);
    p.push_back(z);
}

Cell::Cell(Shape shape, unsigned int i){
    p.push_back(i % shape.dim[0]);
    p.push_back(floor((i % (shape.dim[0] * shape.dim[1])) / shape.dim[1]));
    p.push_back(floor(i / (shape.dim[0] * shape.dim[1])));
}

//---Copy Constructors---//
Cell::Cell(const Cell &obj){
    for(unsigned int i = 0; i < obj.p.size(); i++){
        p.push_back(obj.p[i]);
    }
}

//---Destructors---//
Cell::~Cell(void){}

//---Get---//
unsigned int get_index(Shape shape, unsigned int x, unsigned int y, unsigned int z){
    return x+(y * shape.dim[0])+(z * shape.dim[0] * shape.dim[1]);
}



unsigned int Cell::get_cell_index(Shape shape){
    return get_index(shape, p[0], p[1], p[2]);
}
