#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <new>
#include <random>
#include <windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32
#define MINIMUM_VALUE -32
#define MAXIMUM_VALUE 32
#define DEFAULT_SIZE 1000

void fillMatrix(double* matrix, size_t size)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(MINIMUM_VALUE, MAXIMUM_VALUE);
    
    size_t n = size * size;
    for (size_t i = 0; i < n; ++i)
        matrix[i] = distribution(generator);
}

float multMatricesOnCPU(double* a, double* b, double* c, size_t size)
{
    LARGE_INTEGER startTime, stopTime, freq;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&startTime);
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            size_t cellIndex = i * size + j;
            c[cellIndex] = 0;
            for (size_t k = 0; k < size; ++k)
                c[cellIndex] += a[i * size + k] * b[k * size + j];
        }
    }
    QueryPerformanceCounter(&stopTime);

    size_t timeDelta = stopTime.QuadPart - startTime.QuadPart;
    return static_cast<float>(timeDelta) / freq.QuadPart;
}

__global__ void multMatricesOnGPUKernel(double* a, double* b, double* c, size_t size)
{
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= size || j >= size)
        return;

    size_t cellIndex = i * size + j;
    c[cellIndex] = 0;
    for (size_t k = 0; k < size; ++k)
        c[cellIndex] += a[i * size + k] * b[k * size + j];
}

__global__ void multMatricesOnGPUSharedKernel(double* a, double* b, double* c, size_t size)
{
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    double sum = 0;

    for (size_t subIndex = 0; subIndex * BLOCK_SIZE < size; ++subIndex)
    {
        __shared__ double subA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double subB[BLOCK_SIZE][BLOCK_SIZE];

        subA[threadIdx.y][threadIdx.x] = 0;
        subB[threadIdx.y][threadIdx.x] = 0;

        size_t jSubA = subIndex * BLOCK_SIZE + threadIdx.x;
        size_t iSubB = subIndex * BLOCK_SIZE + threadIdx.y;

        if (i < size && jSubA < size)
            subA[threadIdx.y][threadIdx.x] = a[i * size + jSubA];

        if (iSubB < size && j < size)
            subB[threadIdx.y][threadIdx.x] = b[iSubB * size + j];

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k)
            sum += subA[threadIdx.y][k] * subB[k][threadIdx.x];

        __syncthreads();
    }

    if (i < size && j < size)
        c[i * size + j] = sum;
}

float multMatricesOnGPU(double* a, double* b, double* c, size_t size, bool useSharedMemory)
{
    double* adev, *bdev, *cdev;
    size_t numBytes = size * size * sizeof(double);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

    cudaMalloc(reinterpret_cast<void**>(&adev), numBytes);
    cudaMalloc(reinterpret_cast<void**>(&bdev), numBytes);
    cudaMalloc(reinterpret_cast<void**>(&cdev), numBytes);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    if (useSharedMemory)
        multMatricesOnGPUSharedKernel<<<blocks, threads>>>(adev, bdev, cdev, size);
    else
        multMatricesOnGPUKernel<<<blocks, threads>>>(adev, bdev, cdev, size);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);

    return gpuTime / 1000.0f;
}

double getMaximumDeviation(double* a, double* b, size_t size)
{
    size_t n = size * size;
    double deviation = 0.0;
    for (size_t i = 0; i < n; ++i)
        deviation = std::max(deviation, std::abs(a[i] - b[i]));
    return deviation;
}

int main(int argc, char* argv[])
{
    size_t size = DEFAULT_SIZE;
    if (argc > 1)
        size = std::strtoumax(argv[1], nullptr, 10);
    std::cout << "Matrices have size " << size << "x" << size << std::endl;

    size_t n = size * size;
    double* a = new double[n];
    double* b = new double[n];
    double* cCPU = new double[n];
    double* cGPU = new double[n];
    double* cGPUShared = new double[n];

    fillMatrix(a, size);
    fillMatrix(b, size);

    float timeCPU = multMatricesOnCPU(a, b, cCPU, size);
    float timeGPU = multMatricesOnGPU(a, b, cGPU, size, false);
    float timeGPUShared = multMatricesOnGPU(a, b, cGPUShared, size, true);

    std::cout << "Elapsed times:" << std::endl;
    std::cout << "CPU: " << timeCPU << " seconds" << std::endl;
    std::cout << "GPU: " << timeGPU << " seconds" << std::endl;
    std::cout << "GPU with shared memory: " << timeGPUShared << " seconds" << std::endl;
    std::cout << "Maximum deviation between results on CPU and GPU equals to " << getMaximumDeviation(cCPU, cGPU, size) << std::endl;
    std::cout << "Maximum deviation between results on CPU and GPU with shared memory using equals to " << getMaximumDeviation(cCPU, cGPUShared, size) << std::endl;

    delete[] a;
    delete[] b;
    delete[] cCPU;
    delete[] cGPU;
    delete[] cGPUShared;

    return 0;
}
