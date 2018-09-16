#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive_sm.h"

namespace StreamCompaction {
    namespace NaiveSM {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int N, int *odata, int *idata){
            extern __shared__ int tmp[];
            int pout = 0, pin = 1;
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N){
                tmp[index] = index > 0 ? idata[index - 1]: 0;
                __syncthreads();
                for (int offset = 1; offset < N; offset *= 2){
                    pout = 1 - pout;
                    pin = 1 - pin;
                    if (index >= offset) tmp[pout * N + index] += tmp[pin * N + index - offset];
                    else tmp[pout * N + index ]  = tmp[pin * N + index];
                    __synthreads();
                }
            }
            odata[index] = temp[pout * N + index];
        }

//        // couldn't figure out a way to exclusive scan at once
//        __global__ void kernInclusiveToExclusive(int N, int *odata, int idata){
//            int index = threadIdx.x + (blockIdx.x * blockDim.x);
//            if (index < N){
//                if (index == 0) odata[index] = 0;
//                else odata[index] = idata[index - 1];
//            }
//        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlockPerGrid((n + blockSize - 1) / blockSize);
            int* dev_in, *dev_out;

            cudaMalloc((void**) &dev_in, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in failed");

            cudaMalloc((void**) &dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_out failed");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy HostToDevice failed");

            timer().startGpuTimer();

            kernNaiveScan <<< fullBlockPerGrid, blockSize >>> (n, dev_out, dev_in);
            checkCUDAError("kernNaiveScan dev_in failed");

            // result now in dev_in
//            kernInclusiveToExclusive<<<fullBlockPerGrid, blockSize>>> (n, dev_out, dev_in);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy DeviceToHost failed");

            cudaFree(dev_in);
            cudaFree(dev_out);

        }
    }
}
