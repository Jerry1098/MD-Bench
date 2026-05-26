/*
 * Copyright (C)  NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of MD-Bench.
 * Use of this source code is governed by a LGPL-3.0
 * license that can be found in the LICENSE file.
 */
#ifndef __GPU_PROFILER_H_
#define __GPU_PROFILER_H_

#ifdef CUDA_TARGET
#if CUDA_TARGET == 0
#include <cuda_profiler_api.h>
#define GPU_PROFILE_START(name) cudaProfilerStart()
#define GPU_PROFILE_STOP()      cudaProfilerStop()
#elif CUDA_TARGET == 1
#include <roctracer/roctx.h>
#define GPU_PROFILE_START(name) roctxRangePush(name)
#define GPU_PROFILE_STOP()      roctxRangePop()
#endif
#else
#define GPU_PROFILE_START(name)
#define GPU_PROFILE_STOP()
#endif

#endif
