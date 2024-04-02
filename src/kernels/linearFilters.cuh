#pragma once

template <typename T>
__global__ void genFilterKernel(unsigned char *inp, unsigned char *out, const uint rows, const uint cols, const uint filterSize, T *filter) {
    const uint row = blockDim.y * blockIdx.y + threadIdx.y;
    const uint col = blockDim.x * blockIdx.x + threadIdx.x;
    
    T tmp = 0;
    if(row < rows && col < cols) {
        for(int i = 0; i < filterSize; i++) {
            for(int j = 0; j < filterSize; j++) {
                tmp += filter[i * filterSize + j] * inp[row *  cols + col];
            }
        }
    }
    out[row * cols + col] = static_cast<u_char>(tmp / (filterSize * filterSize));
}