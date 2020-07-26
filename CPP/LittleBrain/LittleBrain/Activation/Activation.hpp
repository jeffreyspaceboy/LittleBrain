//
//  Activation.hpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#ifndef Activation_hpp
#define Activation_hpp

#include <stdio.h>
#include <math.h>

#include "Cell.hpp"

float singleSigmoid(float val, Cell cell);
float singleDSigmoid(float val, Cell cell);

#endif /* Activation_hpp */
