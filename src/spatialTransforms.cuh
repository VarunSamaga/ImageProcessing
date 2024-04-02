#pragma once

#include "kernels/linearFilters.cuh"
#include "utils.cuh"

template<typename T>
void genFilter(unsigned char *img, unsigned char *res, const uint rows, const uint cols, T *filter_h, const int filterSize = 3, const uint channels = 1) {
    unsigned char *inp, *out;
    T *filter;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    
    // cudaMemcpyToSymbol(filter, filter_h, sizeof(T) * filterSize * filterSize);
    cudaMalloc((void**)&filter, sizeof(T) * filterSize * filterSize);

    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    cudaMemcpy(filter, filter_h, sizeof(T) * filterSize * filterSize, cudaMemcpyHostToDevice);

    dim3 blocks(16, 16); 
    dim3 grid(CEIL_DIV(cols * channels, 16), CEIL_DIV(rows, 16));

    genFilterKernel<int><<<grid, blocks>>>(inp, out, rows, cols * channels, filterSize, filter);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}