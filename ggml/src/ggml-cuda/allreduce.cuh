#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Opaque 3-device copy-engine sync state -- dedicated staging buffers and
// events for the large-tensor 3-GPU AllReduce.  Created once per comm context.
struct ggml_cuda_ar3_sync;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// Allocate the 3-device copy-engine sync state.  devices[] holds the three
// CUDA device IDs in rank order; buf_bytes is the per-buffer staging size.
// Returns nullptr on allocation failure.
ggml_cuda_ar3_sync * ggml_cuda_ar3_sync_init(const int * devices, size_t buf_bytes);

// Release the 3-device copy-engine sync state.
void ggml_cuda_ar3_sync_free(ggml_cuda_ar3_sync * sync);

// Three-GPU AllReduce.  Large F32 tensors use the dedicated 3-device
// copy-engine path (requires ar3_sync); small tensors and non-F32 inputs fall
// back to two 2-GPU pipelines:
//   AR(dev0, dev1) -> AR(dev0, dev2) -> copy dev0 -> dev1
// tensors[i] must live on the device managed by backends[i], contiguous
// F32/F16/BF16, same preconditions as ggml_cuda_ar_allreduce.
bool ggml_cuda_ar_allreduce3(
    ggml_cuda_ar_pipeline * pipeline_a,
    ggml_cuda_ar_pipeline * pipeline_b,
    ggml_cuda_ar3_sync    * ar3_sync,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

