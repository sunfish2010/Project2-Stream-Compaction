#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int *odata, int d){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N && index < (1 << (d + 1))){
                odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
            }
        }


        __global__ void kernDownSweep(int N, int *odata, int d){
             int index = threadIdx.x + (blockIdx.x * blockDim.x);
             if (index < N && index < (1 << (d + 1))) {
                int tmp = odata[index + (1 << d) - 1];
                odata[index + (1 << d) - 1]; = odata[index + (1 << (d + 1)) - 1];
                odata[index + (1 << (d + 1)) - 1] += tmp;
             }

        }

        __global__ void kernZeroPad(int N, int n, int *odata){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N && index >= n){
                odata[index] = 0;
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // when n is not power of two, need to allocate more space to zero pad
            int d = ilog2ceil(n);
            int N = 1 << d;

            dim3 fullBlockPerGrid((N + blockSize - 1) / blockSize);
            int* dev_out;

            dev_out = cudaMalloc((void**)&dev_out, sizeof(int) * N);
            checkCUDAError("cudaMalloc dev_out failed");

            cudaMemcpy(dev_out, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpyHostToDevice failed");

            kernZeroPad<<<fullBlockPerGrid, blockSize>>> (N, n, dev_out);
            checkCUDAError("kernZeroPad failed");

            timer().startGpuTimer();
            // without shared memory, the algorithm needs to be called for d times
            for (int i = 0; i < d; i++){
                kernUpSweep<<<fullBlockPerGrid, blockSize>>>(N, dev_out, i);
                checkCUDAError("kernUpSweep failed");
            }

            for (int i = 0; i < d; i++){
                kernDownSweep<<<fullBlockPerGrid, blockSize>>>(N,dev_out, i);
                checkCUDAError("kernDownpSweep failed");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpyDeviceToHost failed");

            cudaFree(odata);
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

            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
