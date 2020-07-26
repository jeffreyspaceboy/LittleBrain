//
//  Cell.hpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#ifndef Cell_hpp
#define Cell_hpp

#include <vector>
#include <math.h>

#include "Shape.hpp"

unsigned int get_index(Shape shape, unsigned int x, unsigned int y, unsigned int z);

class Cell{
    public:
        //---Constructors---//
        Cell(unsigned int x = 0, unsigned int y = 0, unsigned int z = 0);
        Cell(Shape shape, unsigned int i);
        //---Copy Constructors---//
        Cell(const Cell &obj);
        //---Destructors---//
        ~Cell(void);
    
        //---Get---//
        unsigned int get_cell_index(Shape shape);
        
        //---Vars--//
        std::vector<unsigned int> p; //Dimentions of Tensor Size
};

#endif /* Cell_hpp */
