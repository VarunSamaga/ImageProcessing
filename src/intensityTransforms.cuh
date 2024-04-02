#pragma once

#include "kernels/contrastStretch.cuh"
#include "kernels/intensitySlicing.cuh"

#include <opencv2/opencv.hpp>

void imageNegative(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const uint L = 255, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    imageNegativeKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, L);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void logTransform(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const int c = 255, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    logTransformKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, c);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void gammaTransform(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const int c = 1, const float gamma = 1, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    gammaTransformKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, c, gamma);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void intensitySliceThresh(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const uint lrange, const uint urange, const int threshVal, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    intensitySliceThreshKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, lrange, urange, threshVal);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void intensitySliceRange(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const uint lrange, const uint urange, const int threshVal, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    intensitySliceRangeKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, lrange, urange, threshVal);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void bitPlaneSlicing(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const uchar bit, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    bitPlaneSlicingKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, bit);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}

void bitMaskImage(unsigned char *img, unsigned char *res, const uint rows, const uint cols, const uchar mask, const uint channels = 1) {
    unsigned char *inp, *out;
    const size_t SIZE_BYTES = sizeof(unsigned char) * rows * cols * channels;
    cudaMalloc((void**) &inp, SIZE_BYTES);
    cudaMalloc((void**) &out, SIZE_BYTES);
    cudaMemcpy(inp, img, SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blocks(256); 
    dim3 grid(96);

    bitMaskKernel<<<grid, blocks>>>(inp, out, rows * cols * channels, mask);

    cudaMemcpy(res, out, SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(out);
}