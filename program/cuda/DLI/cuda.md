# Quick Start
kernel function
thread structure (grid size & block size, 1D, 2D, 3D)
unified memory malloc and free
device synchronize

# Function Decorator
* `__global__`
* `__host__`
* `__device__`

# Thread Structure
* block-thread structure. A grid contains some blocks and a block contains some threads.
* `gridDim`
* `blockDim`
* global thread idx calculation for 1D case (for both gridDim and blockDim)
* global thread idx calculation for 2D case (for both gridDim and blockDim)
* grid-stride loops

# CUDA Error Catch
* some CUDA runtime APIs return cudaError_t directly.
* cudaGetLastError()->cudaError_t
* cudaGetErrorString(cudaError_t)->const char *

# device properties
* cudaGetDeviceProperties(cudaDeviceProp*, int) obtains properties of device with given device id.
* `cudaDeviceProp.major`
* `cudaDeviceProp.minor`
* `cudaDeviceProp.multiProcessorCount`

# Nsight system
nsys profile

# Data Malloc / Transfer / Free
* `cudaMalloc` `cudaFree` `cudaMemcpy`
* `cudaMallocHost` `cudaFreeHost` `cudaMemcpyAsync` uses given stream. pinned memory
* `cudaMallocManaged` `cudaMemPrefetchAsync` uses given stream. unified memory




# CUDA streams
* stream: a series of operations that occur in issue order on the GPU. Operations in the same stream will execute in issue order.
* non-default stream: operations launched in different non-default streams have no fixed order of execution
* default stream: there can be no execution in any non-default streams at the same time as any execution in the default stream
* stream creation and destrction of non-default stream
`cudaStreamCreate(cudaStream_t *)`
`cudaStreamDestroy(cudaStream_t)`

* CUDA runtime APIs
* kernel functions
4-th launch configuration argument


# copy/compute overlap
partition data into chunks

# Multiple GPUs
obtain the number of available GPUs: `cudaGetDeviceCount(int*)`
obtain the currently active GPU device: `cudaGetDevice(int*)`
set a specific GPU as active: `cudaSetDevice(int)`