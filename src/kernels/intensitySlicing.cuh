#pragma once

__global__ void intensitySliceThreshKernel(u_char *inp, u_char *out, const uint N, const int lrange, const int urange, const uint val) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint stride = gridDim.x * blockDim.x;
    
    while(idx < N) {
        const uint pixVal = inp[idx];
        if(pixVal <= urange && pixVal >= lrange) {
            out[idx] = val;
        } else {
            out[idx] = pixVal;
        }
        idx += stride;
    }
}

__global__ void intensitySliceRangeKernel(u_char *inp, u_char *out, const uint N, const int lrange, const int urange, const uint val) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint stride = gridDim.x * blockDim.x;
    
    while(idx < N) {
        const uint pixVal = inp[idx];
        if(lrange <= pixVal && pixVal <= urange ) {
            out[idx] = val;
        } else {
            out[idx] = pixVal;
        }
        idx += stride;
    }
}

__global__ void bitPlaneSlicingKernel(u_char *inp, u_char *out, const uint N, const unsigned char bit) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint stride = gridDim.x * blockDim.x;
    
    while(idx < N) {
        out[idx] = (inp[idx] & (1 << bit)) ? 255 : 0;
        idx += stride;
    }
}

__global__ void bitMaskKernel(u_char *inp, u_char *out, const uint N, const unsigned char mask) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint stride = gridDim.x * blockDim.x;

    while(idx < N) {
        out[idx] = inp[idx] & mask;
        idx += stride;
    }
}