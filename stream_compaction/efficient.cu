#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int *odata, int d){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            // blocksize is changing 
            if (index < (N >> (d + 1)) ){
                int idx = index << (d + 1);
                odata[idx + (1 << (d + 1)) - 1] += odata[idx + (1 << d) - 1];
            }
        }


        __global__ void kernDownSweep(int N, int *odata, int d){
             int index = threadIdx.x + (blockIdx.x * blockDim.x);
             if (index < (N >> (d + 1)) ) {
                int idx = index << (d + 1);
                int tmp = odata[idx + (1 << d) - 1];
                odata[idx + (1 << d) - 1] = odata[idx + (1 << (d + 1)) - 1];
                odata[idx + (1 << (d + 1)) - 1] += tmp;
             }

        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // when n is not power of two, need to allocate more space to zero pad
            int d = ilog2ceil(n);
            int N = 1 << d;
            int timer_started = 0;

            dim3 fullBlockPerGrid;
            int* dev_out;

            cudaMalloc((void**)&dev_out, sizeof(int) * N);
            checkCUDAError("cudaMalloc dev_out failed");

            cudaMemset(dev_out, 0, sizeof(int) * N);
            checkCUDAError("cuda Memset failed");

            cudaMemcpy(dev_out, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpyHostToDevice failed");

            try
            {
                timer().startGpuTimer();
            }
            catch(...)
            {
                // timer already started 
                timer_started = 1;
            }
            
            // without shared memory, the algorithm needs to be called for d times
            for (int i = 0; i < d; i++){
                fullBlockPerGrid = ((1 << (d - i - 1)) + blockSize - 1) / blockSize;
                kernUpSweep<<<fullBlockPerGrid, blockSize>>>(N, dev_out, i);
                checkCUDAError("kernUpSweep failed");
            }

            cudaMemset(dev_out + N - 1, 0, sizeof(int));
            for (int i = d - 1; i >= 0; i--){
                fullBlockPerGrid = ((1 << (d - i - 1)) + blockSize - 1) / blockSize;
                kernDownSweep<<<fullBlockPerGrid, blockSize>>>(N, dev_out, i);
                checkCUDAError("kernDownpSweep failed");
            }

            if (!timer_started) timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpyDeviceToHost failed");

            cudaFree(dev_out);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            dim3 fullBlockPerGrid((n + blockSize - 1) / blockSize);
            int* bools, *indices, *dev_in, *dev_out;
            int num_element;

            cudaMalloc((void**)&bools, sizeof(int) * n);
            checkCUDAError("cudaMalloc bools failed");
            cudaMalloc((void**)&indices, sizeof(int) * n);
            checkCUDAError("cudaMalloc indices failed");
            cudaMalloc((void**)&dev_out, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_out failed");
            cudaMalloc((void**)&dev_in, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_in failed");

            // lots of memcpy...

            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpyHostToDevice failed");

            timer().startGpuTimer();
            StreamCompaction::Common:: kernMapToBoolean<<<fullBlockPerGrid, blockSize>>>(n, bools, dev_in);
            checkCUDAError("kernMapToBoolean failed");

            cudaMemcpy(odata, bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
            num_element = odata[n - 1];
            checkCUDAError("cudaMemcpyDeviceToHost failed");

            scan(n, odata, odata);
            num_element += odata[n - 1];

            cudaMemcpy(indices, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpyHostToDevice failed");

            StreamCompaction::Common::kernScatter<<<fullBlockPerGrid, blockSize>>>(n, dev_out, dev_in, bools, indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpyDeviceToHost failed");

            cudaFree(bools);
            cudaFree(indices);
            cudaFree(dev_in);
            cudaFree(dev_out);

            return num_element;
        }
    }
}
