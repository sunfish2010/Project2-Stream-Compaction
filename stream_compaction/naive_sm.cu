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

        // naive scan implemented with shared memory

        __global__ void kernNaiveScan(int N, int *odata, int *idata){
			extern __shared__ int tmp[];
			int pout = 0;
			int pin = 1;
            int index = threadIdx.x;
			if (index >= N) return;
            tmp[index] = index > 0 ? idata[index - 1]: 0;
            __syncthreads();
            for (int offset = 1; offset < N; offset *= 2){
                pout = 1 - pout;
                pin = 1 - pin;
                // the suedo code on gems 3 contains error
                if (index >= offset) tmp[pout * N + index] = tmp[pin * N + index - offset] + tmp[pin * N + index];
                else tmp[pout * N + index ]  = tmp[pin * N + index];
                __syncthreads();
            }
        
            odata[index] = tmp[pout * N + index];
        }

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

            kernNaiveScan <<< fullBlockPerGrid, blockSize, 2 * n * sizeof(int) >>> (n, dev_out, dev_in);
            checkCUDAError("kernNaiveScan dev_in failed");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy DeviceToHost failed");

            cudaFree(dev_in);
            cudaFree(dev_out);

        }
    }
}
