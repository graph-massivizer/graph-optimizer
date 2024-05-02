#ifndef DATASTRUCTURES_HPP
#define DATASTRUCTURES_HPP

template<typename T>
class CArray {
public:
    T* data;
    size_t size;

    CArray(size_t size) {
        this->data = new T[size];
        this->size = size;
    }
};

#endif