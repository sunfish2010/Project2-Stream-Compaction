set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
    "naive.h"
    "naive.cu"
    "efficient.h"
    "efficient.cu"
    "thrust.h"
    "thrust.cu"
    "naive_sm.cu"
    "naive_sm.h"
    "efficient_sm.cu"
    "efficient_sm.h"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
