#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

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

// Three-GPU AllReduce composed from two 2-GPU pipelines:
//   AR(dev0, dev1) -> AR(dev0, dev2) -> copy dev0 -> dev1
// dev0 (the pivot) participates in both pairwise reductions.  tensors[i] must
// live on the device managed by backends[i], contiguous F32/F16/BF16, same
// preconditions as ggml_cuda_ar_allreduce.
bool ggml_cuda_ar_allreduce3(
    ggml_cuda_ar_pipeline * pipeline_a,
    ggml_cuda_ar_pipeline * pipeline_b,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// Three-GPU ring AllReduce (copy-engine path): (N-1)-step reduce-scatter +
// allgather, 2*(N-1)/N = 4/3 tensor of per-GPU traffic vs 5 full tensors for
// the two-pipeline composition.  Targets bandwidth-bound (large) tensors;
// the caller keeps small (latency-bound) tensors on the two-pipeline path.
// Same preconditions as ggml_cuda_ar_allreduce; pipeline must be a
// 3-device pipeline.
bool ggml_cuda_ar_allreduce_ring(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// Largest nbytes the ring path can handle in one call: each of the n chunks
// must fit in the per-device copy-engine staging (copy_bytes).  Conservative
// (assumes 4-byte elements).  Tensors above this should use the two-pipeline
// path, which outer-chunks large reductions.
size_t ggml_cuda_ar_ring_max_bytes(ggml_cuda_ar_pipeline * pipeline);

