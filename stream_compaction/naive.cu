#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int N, int *odata, int *idata, int d){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N){
                    if (index >= (1 << d)) odata[index] = idata[index - (1 << d)] + idata[index];
                    else odata[index] = idata[index];
                }
        }

        // couldn't figure out a way to exclusive scan at once
        __global__ void kernInclusiveToExclusive(int N, int *odata, int *idata){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N){
                if (index == 0) odata[index] = 0;
                else odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlockPerGrid((n + blockSize - 1) / blockSize);
            int* dev_in, *dev_out;
            // int out = 0;

            cudaMalloc((void**) &dev_in, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in failed");

            cudaMalloc((void**) &dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_out failed");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy HostToDevice failed");

            // cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // checkCUDAError("cudaMemcpy HostToDevice failed");

            timer().startGpuTimer();
            // we don't need to allocate more mem space since the algorithm is never accessing space > n
            for (int d = 0; d < ilog2ceil(n); d++) {
                // ping-pong the buffer for 'inplace' matrix manipulation
                kernNaiveScan <<< fullBlockPerGrid, blockSize >>> (n, dev_out, dev_in, d);
                checkCUDAError("kernNaiveScan dev_in failed");
                std::swap(dev_in, dev_out);
            }
            // result now in dev_in
            kernInclusiveToExclusive<<<fullBlockPerGrid, blockSize>>> (n, dev_out, dev_in);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy DeviceToHost failed");

            cudaFree(dev_in);
            cudaFree(dev_out);

        }
    }
}
