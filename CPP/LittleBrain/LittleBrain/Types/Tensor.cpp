//
//  Tensor.cpp
//  LittleBrain
//
//  Created by Jeffrey Fisher on 7/25/20.
//  Copyright © 2020 Jeffrey Fisher. All rights reserved.
//

#include "Tensor.hpp"

//---Constructors---//
Tensor::Tensor(void){} //Blank Constructor

Tensor::Tensor(std::vector<float> new_data, Shape shape){ //Standard Constructor
    set_data(new_data, shape);
}
Tensor::Tensor(std::vector<float> new_data){
    set_data(new_data, Shape((unsigned int)new_data.size()));
}
Tensor::Tensor(float *new_data, Shape shape){
    for(unsigned int i = 0; i < shape.size; i++){
        data.push_back(new_data[i]);
    }
}

Tensor::Tensor(Shape shape, bool do_randomize){ //Zero or Randomize Constructor
    if(do_randomize){
        set_shape(shape);
        randomize(-1, 1, 1000);
    }else{
        set_data(0.0, shape);
    }
}

//---Copy Constructors---//
Tensor::Tensor(const Tensor &obj){ set_data(obj.data, obj.shape); }


//---Destructors---//
Tensor::~Tensor(void){}


//---Set---//
void Tensor::set_shape(Shape new_shape){
    if(!data.empty()){
        printf("WARNING: This matrix contains data! The data may be lost if shape does not match the original...\n");
    }
    shape = new_shape;
}

void Tensor::set_data(std::vector<float> new_data, Shape shape){
    if(!check_shape(shape)){
        set_shape(shape);
    }
    if(!data.empty()){
        printf("WARNING: This matrix contains data! The data is being cleared...\n");
        data.clear();
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data.push_back(new_data[i]);
    }
}

void Tensor::set_data(float new_data, Shape shape){
    if(!check_shape(shape)){
        set_shape(shape);
    }
    if(!this->data.empty()){
        printf("WARNING: This matrix contains data! The data is being cleared...\n");
        data.clear();
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data.push_back(new_data);
    }
}

void Tensor::set_cell(float new_data, Cell cell){
    data[cell.get_cell_index(shape)] = new_data;
}

void Tensor::set_cell(float new_data, unsigned int cell){
    data[cell] = new_data;
}


//---Get---//
Shape Tensor::get_shape(void){
    return shape;
}

std::vector<float> Tensor::get_data(void){
    return data;
}

float Tensor::get_cell(unsigned int cell){
    return data[cell];
}

float Tensor::get_cell(Cell cell){
    return data[cell.get_cell_index(shape)];
}

Tensor Tensor::get_matrix(unsigned int z){
    std::vector<float> matrix_data;
    unsigned int i_start = get_index(shape,0,0,z);
    for(unsigned int i = i_start; i < (i_start + (shape.dim[0]*shape.dim[1])); i++){
        matrix_data.push_back(data[i]);
    }
    return Tensor(matrix_data, Shape(shape.dim[0],shape.dim[1]));
}

//---Math Operations---//
void Tensor::map(float (*func)(float val, Cell cell)){
    for(unsigned int i = 0; i < shape.size; i++){
        float val = data[i];
        data[i] = (*func)(val, Cell(shape,i));
    }
}

Tensor *Tensor::map(Tensor *a, float (*func)(float val, Cell cell)){
    for(unsigned int i = 0; i < a->shape.size; i++){
        float val = a->data[i];
        a->data[i] = (*func)(val, Cell(a->shape,i));
    }
    return a;
}


void Tensor::add(Tensor &obj){
    if(!check_shape(obj.shape)){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] + obj.data[i];
    }
}

void Tensor::add(float obj){
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] + obj;
    }
}

void Tensor::subtract(Tensor &obj){
    if(!(this->check_shape(obj.shape))){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] - obj.data[i];
    }
}

void Tensor::subtract(float obj){
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] - obj;
    }
}

void Tensor::matrix_product(Tensor &obj){
    if(shape.dim[0] != obj.shape.dim[1]){
        printf("ERROR: For cross product columns of A must match rows of B.\n");
        exit(1);
    }
    if(shape.dim[2] != obj.shape.dim[2]){
        printf("ERROR: Layers must match.\n");
        exit(1);
    }
    Tensor result(Shape(obj.shape.dim[0], shape.dim[1], shape.dim[2]));
    Cell cell;
    unsigned int z;
    float sum;
    for(cell.p[2] = 0; cell.p[2] < result.shape.dim[2]; cell.p[2]++){
        for(cell.p[1] = 0; cell.p[1] < result.shape.dim[1]; cell.p[1]++){
            for(cell.p[0] = 0; cell.p[0] < result.shape.dim[0]; cell.p[0]++){
                sum = 0;
                for(z = 0; z < shape.dim[0]; z++){
                    sum += data[get_index(shape, z, cell.p[1], cell.p[2])] * obj.data[get_index(result.shape, cell.p[0], z, cell.p[2])];
                    //sum += this->data[z + (cell.p[1] * this->shape.dim[0]) + (cell.p[2] * this->shape.dim[0] * this->shape.dim[1])] * obj.data[cell.p[0] + (z * result.shape.dim[0]) + (cell.p[2] * this->shape.dim[0] * result.shape.dim[0])];
                }
                result.set_cell(sum,cell);
            }
        }
    }
    shape = result.shape;
    data = result.data;
}

void Tensor::hadamard_product(Tensor &obj){
    if(!check_shape(obj.shape)){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] * obj.data[i];
    }
}

void Tensor::scalar_product(float obj){
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = data[i] * obj;
    }
}

Tensor Tensor::operator +(Tensor &obj){ //Overloading add operator
    if(!check_shape(obj.shape)){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] + obj.data[i]);
    }
    return Tensor(new_data,shape);
}

Tensor Tensor::operator +(float obj){ //Overloading add operator
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] + obj);
    }
    return Tensor(new_data, shape);
}

Tensor Tensor::operator -(Tensor &obj){ //Overloading subtract operator
    if(!check_shape(obj.shape)){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] - obj.data[i]);
    }
    return Tensor(new_data, shape);
}

Tensor Tensor::operator -(float obj){ //Overloading subtract operator
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] - obj);
    }
    return Tensor(new_data, shape);
}

Tensor Tensor::operator *(Tensor &obj){ //CROSS PRODUCT - overloading the * operator
    if(shape.dim[0] != obj.shape.dim[1]){
        printf("ERROR: For cross product columns of A must match rows of B.\n");
        exit(1);
    }
    if(shape.dim[2] != obj.shape.dim[2]){
        printf("ERROR: Layers must match.\n");
        exit(1);
    }
    Tensor result(Shape(obj.shape.dim[0], shape.dim[1], shape.dim[2]));
    Cell cell;
    unsigned int z;
    float sum;
    for(cell.p[2] = 0; cell.p[2] < result.shape.dim[2]; cell.p[2]++){
        for(cell.p[1] = 0; cell.p[1] < result.shape.dim[1]; cell.p[1]++){
            for(cell.p[0] = 0; cell.p[0] < result.shape.dim[0]; cell.p[0]++){
                sum = 0;
                for(z = 0; z < shape.dim[0]; z++){
                    sum += data[get_index(shape, z, cell.p[1], cell.p[2])] * obj.data[get_index(result.shape, cell.p[0], z, cell.p[2])];
                    //sum += this->data[z + (cell.p[1] * this->shape.dim[0]) + (cell.p[2] * this->shape.dim[0] * this->shape.dim[1])] * obj.data[cell.p[0] + (z * result.shape.dim[0]) + (cell.p[2] * this->shape.dim[0] * result.shape.dim[0])];
                }
                result.set_cell(sum,cell);
            }
        }
    }
    return result;
}

Tensor Tensor::operator ->*(Tensor &obj){ //HADAMARD PRODUCT - overloading the ->* (not sure what that is normally?) operator
    if(!check_shape(obj.shape)){
        printf("ERROR: Shape of matrices must match.\n");
        exit(1);
    }
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] * obj.data[i]);
    }
    return Tensor(new_data, shape);
}

Tensor Tensor::operator *(float obj){ //SCALAR PRODUCT - overloading the * operator
    std::vector<float> new_data;
    for(unsigned int i = 0; i < shape.size; i++){
        new_data.push_back(data[i] * obj);
    }
    return Tensor(new_data, shape);
}

Tensor Tensor::operator ~(void){ return transposed(); }//TRANSPOSE - overloading the ~ operator

Tensor Tensor::transposed(void){ //TRANSPOSE
    Cell cell;
    Tensor result(Shape(shape.dim[1], shape.dim[0], shape.dim[2]));
    for(cell.p[2] = 0; cell.p[2] < result.shape.dim[2]; cell.p[2]++){
        for(cell.p[1] = 0; cell.p[1] < result.shape.dim[1]; cell.p[1]++){
            for(cell.p[0] = 0; cell.p[0] < result.shape.dim[0]; cell.p[0]++){
                result.set_cell(data[get_index(shape, cell.p[1], cell.p[0], cell.p[2])], cell);
            }
        }
    }
    return result;
}

float Tensor::random(int lowerBound, int upperBound, int decimal_precision){
    lowerBound = lowerBound*decimal_precision;
    upperBound = upperBound*decimal_precision;
    return ((float)(rand() % (upperBound-lowerBound+1) + lowerBound)/decimal_precision);
}

void Tensor::randomize(int lowerBound, int upperBound, int decimal_precision){
    if(!data.empty()){
        printf("WARNING: This matrix contains data! The data is being cleared...\n");
        data.clear();
    }
    for(unsigned int i = 0; i < shape.size; i++){
        data.push_back(this->random(lowerBound, upperBound, decimal_precision));
    }
}

void Tensor::round_to(float val){
    for(unsigned int i = 0; i < shape.size; i++){
        data[i] = floor(data[i] * val + 0.5)/val;
    }
}

//---Activation Functions---//
void Tensor::sigmoid(void){ map(singleSigmoid); }
void Tensor::dSigmoid(void){ map(singleDSigmoid); }


//---Checking---//
bool Tensor::check_matrix(void){
    if(shape.size <= 0){
        printf("ERROR: No matrix exists.");
        return false;
    }
    return true;
}

bool Tensor::check_shape(Shape shape){
    if((this->shape.dim[0] != shape.dim[0])||(this->shape.dim[1] != shape.dim[1])||(this->shape.dim[2] != shape.dim[2])){ return false; }
    else { return true; }
}

//---Other---//
void Tensor::print(void){
    Cell cell;
    printf("{\n");
    for(cell.p[2] = 0; cell.p[2] < shape.dim[2]; cell.p[2]++) {
        printf("  Layer: %d\n", cell.p[2]);
        for(cell.p[1] = 0; cell.p[1] < shape.dim[1]; cell.p[1]++) {
            printf("  [");
            for(cell.p[0] = 0; cell.p[0] < shape.dim[0]-1; cell.p[0]++){
                printf("%f,",data[cell.get_cell_index(shape)]);
            }
            printf("%f]\n",data[get_index(shape, (shape.dim[0]-1), cell.p[1], cell.p[2])]);
        }
    }
    printf("}\n");
}

