#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {

__device__ __forceinline__ float atomic_max_float(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i;
  int assumed;
  while (val > __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    if (assumed == old) {
      break;
    }
  }
  return __int_as_float(old);
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float dropout_scale_for(
    std::uint64_t seed,
    int64_t linear_index,
    float dropout_p) {
  if (dropout_p <= 0.0f) {
    return 1.0f;
  }
  std::uint64_t bits = splitmix64(seed ^ static_cast<std::uint64_t>(linear_index));
  float u = static_cast<float>((bits >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
  return (u >= dropout_p) ? (1.0f / (1.0f - dropout_p)) : 0.0f;
}

template <typename scalar_t>
__device__ __forceinline__ float read_scalar(const scalar_t* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <typename scalar_t>
__device__ __forceinline__ float relation_project_value(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int64_t edge,
    int64_t hd_index,
    int64_t relation_dim,
    bool has_bias) {
  float value = has_bias ? read_scalar(bias, hd_index) : 0.0f;
  int64_t r_base = edge * relation_dim;
  int64_t weight_base = hd_index * relation_dim;
  for (int64_t rd = 0; rd < relation_dim; ++rd) {
    value += read_scalar(r, r_base + rd) * read_scalar(weight, weight_base + rd);
  }
  return value;
}

template <typename scalar_t>
__global__ void max_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_edge,
    const int64_t* __restrict__ sorted_dst,
    float* __restrict__ max_score,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    float scale) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];

  float score = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t k_base = (edge * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    score += read_scalar(q, q_base + d) * read_scalar(k_edge, k_base + d);
  }
  score *= scale;
  atomic_max_float(max_score + dst * num_heads + head, score);
}

template <typename scalar_t>
__global__ void max_direct_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ relation_key_weight,
    const int64_t* __restrict__ sorted_src,
    const int64_t* __restrict__ sorted_dst,
    float* __restrict__ max_score,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    int64_t relation_dim,
    float scale) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];
  int64_t src = sorted_src[edge];

  float score = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t k_base = (src * num_heads + head) * head_dim;
  int64_t hd_base = head * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float key_rel = relation_project_value(
        r,
        relation_key_weight,
        relation_key_weight,
        edge,
        hd_index,
        relation_dim,
        false);
    score += read_scalar(q, q_base + d) * (read_scalar(k, k_base + d) + key_rel);
  }
  score *= scale;
  atomic_max_float(max_score + dst * num_heads + head, score);
}

template <typename scalar_t>
__global__ void sum_output_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_edge,
    const scalar_t* __restrict__ v_edge,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ max_score,
    float* __restrict__ denom,
    float* __restrict__ out_acc,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];

  float score = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (edge * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    score += read_scalar(q, q_base + d) * read_scalar(k_edge, kv_base + d);
  }
  score *= scale;
  float exp_score = expf(score - max_score[dst * num_heads + head]);
  atomicAdd(denom + dst * num_heads + head, exp_score);

  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float weighted = exp_score * drop_scale;
  int64_t out_base = (dst * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    atomicAdd(out_acc + out_base + d, weighted * read_scalar(v_edge, kv_base + d));
  }
}

template <typename scalar_t>
__global__ void sum_output_direct_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ relation_key_weight,
    const scalar_t* __restrict__ relation_value_weight,
    const scalar_t* __restrict__ relation_value_bias,
    const int64_t* __restrict__ sorted_src,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ max_score,
    float* __restrict__ denom,
    float* __restrict__ out_acc,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    int64_t relation_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed,
    bool has_value_bias) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];
  int64_t src = sorted_src[edge];

  float score = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (src * num_heads + head) * head_dim;
  int64_t hd_base = head * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float key_rel = relation_project_value(
        r,
        relation_key_weight,
        relation_key_weight,
        edge,
        hd_index,
        relation_dim,
        false);
    score += read_scalar(q, q_base + d) * (read_scalar(k, kv_base + d) + key_rel);
  }
  score *= scale;
  float exp_score = expf(score - max_score[dst * num_heads + head]);
  atomicAdd(denom + dst * num_heads + head, exp_score);

  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float weighted = exp_score * drop_scale;
  int64_t out_base = (dst * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float value_rel = relation_project_value(
        r,
        relation_value_weight,
        relation_value_bias,
        edge,
        hd_index,
        relation_dim,
        has_value_bias);
    atomicAdd(out_acc + out_base + d, weighted * (read_scalar(v, kv_base + d) + value_rel));
  }
}

template <typename scalar_t>
__global__ void normalize_kernel(
    const float* __restrict__ max_score,
    const float* __restrict__ denom,
    const float* __restrict__ out_acc,
    scalar_t* __restrict__ out,
    float* __restrict__ lse,
    int64_t num_dst,
    int64_t num_heads,
    int64_t head_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_dst * num_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int64_t d = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_heads;
  int64_t dst = tmp / num_heads;
  int64_t nh = dst * num_heads + head;
  float den = denom[nh];
  float value = den > 0.0f ? out_acc[idx] / den : 0.0f;
  out[idx] = static_cast<scalar_t>(value);
  if (d == 0) {
    lse[nh] = den > 0.0f ? max_score[nh] + logf(den) : -INFINITY;
  }
}

template <typename scalar_t>
__global__ void group_dot_direct_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ relation_key_weight,
    const scalar_t* __restrict__ relation_value_weight,
    const scalar_t* __restrict__ relation_value_bias,
    const int64_t* __restrict__ sorted_src,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ lse,
    float* __restrict__ group_dot,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    int64_t relation_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed,
    bool has_value_bias) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];
  int64_t src = sorted_src[edge];

  float score = 0.0f;
  float grad_attn_used = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (src * num_heads + head) * head_dim;
  int64_t hd_base = head * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float key_rel = relation_project_value(
        r,
        relation_key_weight,
        relation_key_weight,
        edge,
        hd_index,
        relation_dim,
        false);
    float value_rel = relation_project_value(
        r,
        relation_value_weight,
        relation_value_bias,
        edge,
        hd_index,
        relation_dim,
        has_value_bias);
    score += read_scalar(q, q_base + d) * (read_scalar(k, kv_base + d) + key_rel);
    grad_attn_used += read_scalar(grad_out, q_base + d) * (read_scalar(v, kv_base + d) + value_rel);
  }
  score *= scale;
  float attn = expf(score - lse[dst * num_heads + head]);
  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float grad_attn = grad_attn_used * drop_scale;
  atomicAdd(group_dot + dst * num_heads + head, attn * grad_attn);
}

template <typename scalar_t>
__global__ void group_dot_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_edge,
    const scalar_t* __restrict__ v_edge,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ lse,
    float* __restrict__ group_dot,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];

  float score = 0.0f;
  float grad_attn_used = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (edge * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    score += read_scalar(q, q_base + d) * read_scalar(k_edge, kv_base + d);
    grad_attn_used += read_scalar(grad_out, q_base + d) * read_scalar(v_edge, kv_base + d);
  }
  score *= scale;
  float attn = expf(score - lse[dst * num_heads + head]);
  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float grad_attn = grad_attn_used * drop_scale;
  atomicAdd(group_dot + dst * num_heads + head, attn * grad_attn);
}

template <typename scalar_t>
__global__ void grad_direct_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ relation_key_weight,
    const scalar_t* __restrict__ relation_value_weight,
    const scalar_t* __restrict__ relation_value_bias,
    const int64_t* __restrict__ sorted_src,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ lse,
    const float* __restrict__ group_dot,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    float* __restrict__ grad_r,
    float* __restrict__ grad_relation_key_weight,
    float* __restrict__ grad_relation_value_weight,
    float* __restrict__ grad_relation_value_bias,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    int64_t relation_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed,
    bool has_value_bias) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];
  int64_t src = sorted_src[edge];

  float score = 0.0f;
  float grad_attn_used = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (src * num_heads + head) * head_dim;
  int64_t hd_base = head * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float key_rel = relation_project_value(
        r,
        relation_key_weight,
        relation_key_weight,
        edge,
        hd_index,
        relation_dim,
        false);
    float value_rel = relation_project_value(
        r,
        relation_value_weight,
        relation_value_bias,
        edge,
        hd_index,
        relation_dim,
        has_value_bias);
    score += read_scalar(q, q_base + d) * (read_scalar(k, kv_base + d) + key_rel);
    grad_attn_used += read_scalar(grad_out, q_base + d) * (read_scalar(v, kv_base + d) + value_rel);
  }
  score *= scale;
  float attn = expf(score - lse[dst * num_heads + head]);
  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float grad_attn = grad_attn_used * drop_scale;
  float grad_score = attn * (grad_attn - group_dot[dst * num_heads + head]);
  float grad_dot = grad_score * scale;
  float attn_used = attn * drop_scale;

  for (int64_t d = 0; d < head_dim; ++d) {
    int64_t hd_index = hd_base + d;
    float q_value = read_scalar(q, q_base + d);
    float key_grad = grad_dot * q_value;
    float value_grad = attn_used * read_scalar(grad_out, q_base + d);

    atomicAdd(grad_q + q_base + d, grad_dot * (
        read_scalar(k, kv_base + d)
        + relation_project_value(
            r,
            relation_key_weight,
            relation_key_weight,
            edge,
            hd_index,
            relation_dim,
            false)));
    atomicAdd(grad_k + kv_base + d, key_grad);
    atomicAdd(grad_v + kv_base + d, value_grad);

    for (int64_t rd = 0; rd < relation_dim; ++rd) {
      int64_t r_index = edge * relation_dim + rd;
      int64_t weight_index = hd_index * relation_dim + rd;
      float r_value = read_scalar(r, r_index);
      atomicAdd(
          grad_r + r_index,
          key_grad * read_scalar(relation_key_weight, weight_index)
              + value_grad * read_scalar(relation_value_weight, weight_index));
      atomicAdd(grad_relation_key_weight + weight_index, key_grad * r_value);
      atomicAdd(grad_relation_value_weight + weight_index, value_grad * r_value);
    }
    if (has_value_bias) {
      atomicAdd(grad_relation_value_bias + hd_index, value_grad);
    }
  }
}

template <typename scalar_t>
__global__ void grad_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_edge,
    const scalar_t* __restrict__ v_edge,
    const int64_t* __restrict__ sorted_dst,
    const float* __restrict__ lse,
    const float* __restrict__ group_dot,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k_edge,
    float* __restrict__ grad_v_edge,
    int64_t num_edges,
    int64_t num_heads,
    int64_t head_dim,
    float scale,
    float dropout_p,
    std::uint64_t seed) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = num_edges * num_heads;
  if (idx >= total) {
    return;
  }
  int64_t edge = idx / num_heads;
  int64_t head = idx - edge * num_heads;
  int64_t dst = sorted_dst[edge];

  float score = 0.0f;
  float grad_attn_used = 0.0f;
  int64_t q_base = (dst * num_heads + head) * head_dim;
  int64_t kv_base = (edge * num_heads + head) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    score += read_scalar(q, q_base + d) * read_scalar(k_edge, kv_base + d);
    grad_attn_used += read_scalar(grad_out, q_base + d) * read_scalar(v_edge, kv_base + d);
  }
  score *= scale;
  float attn = expf(score - lse[dst * num_heads + head]);
  float drop_scale = dropout_scale_for(seed, edge * num_heads + head, dropout_p);
  float grad_attn = grad_attn_used * drop_scale;
  float grad_score = attn * (grad_attn - group_dot[dst * num_heads + head]);
  float grad_dot = grad_score * scale;
  float attn_used = attn * drop_scale;

  for (int64_t d = 0; d < head_dim; ++d) {
    float q_value = read_scalar(q, q_base + d);
    float k_value = read_scalar(k_edge, kv_base + d);
    float grad_out_value = read_scalar(grad_out, q_base + d);
    atomicAdd(grad_q + q_base + d, grad_dot * k_value);
    grad_k_edge[kv_base + d] = grad_dot * q_value;
    grad_v_edge[kv_base + d] = attn_used * grad_out_value;
  }
}

}  // namespace

std::vector<torch::Tensor> segmented_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_edge,
    torch::Tensor v_edge,
    torch::Tensor sorted_dst,
    torch::Tensor dst_ptr,
    double scale,
    double dropout_p,
    std::uint64_t seed) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_edge);
  CHECK_INPUT(v_edge);
  CHECK_INPUT(sorted_dst);
  CHECK_INPUT(dst_ptr);
  TORCH_CHECK(q.dim() == 3, "q must have shape [N_dst, H, D]");
  TORCH_CHECK(k_edge.dim() == 3, "k_edge must have shape [E, H, D]");
  TORCH_CHECK(v_edge.sizes() == k_edge.sizes(), "v_edge shape must match k_edge");
  TORCH_CHECK(q.size(1) == k_edge.size(1) && q.size(2) == k_edge.size(2), "q and k_edge must share H and D");
  TORCH_CHECK(sorted_dst.scalar_type() == torch::kLong, "sorted_dst must be int64");
  TORCH_CHECK(dst_ptr.scalar_type() == torch::kLong, "dst_ptr must be int64");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  int64_t num_dst = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_edges = k_edge.size(0);

  auto float_options = q.options().dtype(torch::kFloat32);
  auto max_score = torch::full({num_dst, num_heads}, -std::numeric_limits<float>::infinity(), float_options);
  auto denom = torch::zeros({num_dst, num_heads}, float_options);
  auto out_acc = torch::zeros({num_dst, num_heads, head_dim}, float_options);
  auto out = torch::empty_like(q);
  auto lse = torch::full({num_dst, num_heads}, -std::numeric_limits<float>::infinity(), float_options);

  if (num_edges > 0) {
    int threads = 256;
    int64_t edge_head_total = num_edges * num_heads;
    int blocks = static_cast<int>((edge_head_total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "segmented_attention_forward_cuda",
        [&] {
      max_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          q.data_ptr<scalar_t>(),
          k_edge.data_ptr<scalar_t>(),
          sorted_dst.data_ptr<int64_t>(),
          max_score.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      sum_output_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          q.data_ptr<scalar_t>(),
          k_edge.data_ptr<scalar_t>(),
          v_edge.data_ptr<scalar_t>(),
          sorted_dst.data_ptr<int64_t>(),
          max_score.data_ptr<float>(),
          denom.data_ptr<float>(),
          out_acc.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      int64_t out_total = num_dst * num_heads * head_dim;
      int out_blocks = static_cast<int>((out_total + threads - 1) / threads);
      normalize_kernel<scalar_t><<<out_blocks, threads, 0, stream>>>(
          max_score.data_ptr<float>(),
          denom.data_ptr<float>(),
          out_acc.data_ptr<float>(),
          out.data_ptr<scalar_t>(),
          lse.data_ptr<float>(),
          num_dst,
          num_heads,
          head_dim);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else {
    out.zero_();
  }
  return {out, lse};
}

std::vector<torch::Tensor> segmented_attention_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k_edge,
    torch::Tensor v_edge,
    torch::Tensor sorted_dst,
    torch::Tensor dst_ptr,
    torch::Tensor lse,
    double scale,
    double dropout_p,
    std::uint64_t seed) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(q);
  CHECK_INPUT(k_edge);
  CHECK_INPUT(v_edge);
  CHECK_INPUT(sorted_dst);
  CHECK_INPUT(dst_ptr);
  CHECK_INPUT(lse);
  TORCH_CHECK(q.dim() == 3, "q must have shape [N_dst, H, D]");
  TORCH_CHECK(k_edge.dim() == 3, "k_edge must have shape [E, H, D]");
  TORCH_CHECK(v_edge.sizes() == k_edge.sizes(), "v_edge shape must match k_edge");
  TORCH_CHECK(grad_out.sizes() == q.sizes(), "grad_out shape must match q");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  int64_t num_dst = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_edges = k_edge.size(0);

  auto float_options = q.options().dtype(torch::kFloat32);
  auto group_dot = torch::zeros({num_dst, num_heads}, float_options);
  auto grad_q = torch::zeros({num_dst, num_heads, head_dim}, float_options);
  auto grad_k_edge = torch::empty({num_edges, num_heads, head_dim}, float_options);
  auto grad_v_edge = torch::empty({num_edges, num_heads, head_dim}, float_options);

  if (num_edges > 0) {
    int threads = 256;
    int64_t edge_head_total = num_edges * num_heads;
    int blocks = static_cast<int>((edge_head_total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "segmented_attention_backward_cuda",
        [&] {
      group_dot_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k_edge.data_ptr<scalar_t>(),
          v_edge.data_ptr<scalar_t>(),
          sorted_dst.data_ptr<int64_t>(),
          lse.data_ptr<float>(),
          group_dot.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      grad_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k_edge.data_ptr<scalar_t>(),
          v_edge.data_ptr<scalar_t>(),
          sorted_dst.data_ptr<int64_t>(),
          lse.data_ptr<float>(),
          group_dot.data_ptr<float>(),
          grad_q.data_ptr<float>(),
          grad_k_edge.data_ptr<float>(),
          grad_v_edge.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return {grad_q, grad_k_edge, grad_v_edge};
}

std::vector<torch::Tensor> segmented_attention_forward_direct_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r,
    torch::Tensor relation_key_weight,
    torch::Tensor relation_value_weight,
    torch::Tensor relation_value_bias,
    torch::Tensor sorted_src,
    torch::Tensor sorted_dst,
    torch::Tensor dst_ptr,
    double scale,
    double dropout_p,
    std::uint64_t seed,
    bool has_value_bias) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(r);
  CHECK_INPUT(relation_key_weight);
  CHECK_INPUT(relation_value_weight);
  CHECK_INPUT(relation_value_bias);
  CHECK_INPUT(sorted_src);
  CHECK_INPUT(sorted_dst);
  CHECK_INPUT(dst_ptr);
  TORCH_CHECK(q.dim() == 3, "q must have shape [N_dst, H, D]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [N_src, H, D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k");
  TORCH_CHECK(r.dim() == 2, "r must have shape [E, R]");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(2) == k.size(2), "q and k must share H and D");
  TORCH_CHECK(sorted_src.scalar_type() == torch::kLong, "sorted_src must be int64");
  TORCH_CHECK(sorted_dst.scalar_type() == torch::kLong, "sorted_dst must be int64");
  TORCH_CHECK(dst_ptr.scalar_type() == torch::kLong, "dst_ptr must be int64");
  TORCH_CHECK(relation_key_weight.dim() == 2, "relation_key_weight must have shape [H*D, R]");
  TORCH_CHECK(relation_value_weight.sizes() == relation_key_weight.sizes(), "relation weights must share shape");
  TORCH_CHECK(relation_key_weight.size(0) == q.size(1) * q.size(2), "relation weight first dimension must be H*D");
  TORCH_CHECK(relation_key_weight.size(1) == r.size(1), "relation weight second dimension must be R");
  TORCH_CHECK(!has_value_bias || relation_value_bias.numel() == q.size(1) * q.size(2), "relation_value_bias must have H*D entries");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  int64_t num_dst = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_edges = r.size(0);
  int64_t relation_dim = r.size(1);

  auto float_options = q.options().dtype(torch::kFloat32);
  auto max_score = torch::full({num_dst, num_heads}, -std::numeric_limits<float>::infinity(), float_options);
  auto denom = torch::zeros({num_dst, num_heads}, float_options);
  auto out_acc = torch::zeros({num_dst, num_heads, head_dim}, float_options);
  auto out = torch::empty_like(q);
  auto lse = torch::full({num_dst, num_heads}, -std::numeric_limits<float>::infinity(), float_options);

  if (num_edges > 0) {
    int threads = 256;
    int64_t edge_head_total = num_edges * num_heads;
    int blocks = static_cast<int>((edge_head_total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "segmented_attention_forward_direct_cuda",
        [&] {
      max_direct_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          r.data_ptr<scalar_t>(),
          relation_key_weight.data_ptr<scalar_t>(),
          sorted_src.data_ptr<int64_t>(),
          sorted_dst.data_ptr<int64_t>(),
          max_score.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          relation_dim,
          static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      sum_output_direct_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          r.data_ptr<scalar_t>(),
          relation_key_weight.data_ptr<scalar_t>(),
          relation_value_weight.data_ptr<scalar_t>(),
          relation_value_bias.data_ptr<scalar_t>(),
          sorted_src.data_ptr<int64_t>(),
          sorted_dst.data_ptr<int64_t>(),
          max_score.data_ptr<float>(),
          denom.data_ptr<float>(),
          out_acc.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          relation_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed,
          has_value_bias);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      int64_t out_total = num_dst * num_heads * head_dim;
      int out_blocks = static_cast<int>((out_total + threads - 1) / threads);
      normalize_kernel<scalar_t><<<out_blocks, threads, 0, stream>>>(
          max_score.data_ptr<float>(),
          denom.data_ptr<float>(),
          out_acc.data_ptr<float>(),
          out.data_ptr<scalar_t>(),
          lse.data_ptr<float>(),
          num_dst,
          num_heads,
          head_dim);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else {
    out.zero_();
  }
  return {out, lse};
}

std::vector<torch::Tensor> segmented_attention_backward_direct_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r,
    torch::Tensor relation_key_weight,
    torch::Tensor relation_value_weight,
    torch::Tensor relation_value_bias,
    torch::Tensor sorted_src,
    torch::Tensor sorted_dst,
    torch::Tensor dst_ptr,
    torch::Tensor lse,
    double scale,
    double dropout_p,
    std::uint64_t seed,
    bool has_value_bias) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(r);
  CHECK_INPUT(relation_key_weight);
  CHECK_INPUT(relation_value_weight);
  CHECK_INPUT(relation_value_bias);
  CHECK_INPUT(sorted_src);
  CHECK_INPUT(sorted_dst);
  CHECK_INPUT(dst_ptr);
  CHECK_INPUT(lse);
  TORCH_CHECK(q.dim() == 3, "q must have shape [N_dst, H, D]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [N_src, H, D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k");
  TORCH_CHECK(grad_out.sizes() == q.sizes(), "grad_out shape must match q");
  TORCH_CHECK(r.dim() == 2, "r must have shape [E, R]");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  int64_t num_dst = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_edges = r.size(0);
  int64_t relation_dim = r.size(1);

  auto float_options = q.options().dtype(torch::kFloat32);
  auto group_dot = torch::zeros({num_dst, num_heads}, float_options);
  auto grad_q = torch::zeros({q.size(0), q.size(1), q.size(2)}, float_options);
  auto grad_k = torch::zeros({k.size(0), k.size(1), k.size(2)}, float_options);
  auto grad_v = torch::zeros({v.size(0), v.size(1), v.size(2)}, float_options);
  auto grad_r = torch::zeros({r.size(0), r.size(1)}, float_options);
  auto grad_relation_key_weight = torch::zeros({relation_key_weight.size(0), relation_key_weight.size(1)}, float_options);
  auto grad_relation_value_weight = torch::zeros({relation_value_weight.size(0), relation_value_weight.size(1)}, float_options);
  auto grad_relation_value_bias = torch::zeros({relation_value_bias.numel()}, float_options);

  if (num_edges > 0) {
    int threads = 256;
    int64_t edge_head_total = num_edges * num_heads;
    int blocks = static_cast<int>((edge_head_total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "segmented_attention_backward_direct_cuda",
        [&] {
      group_dot_direct_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          r.data_ptr<scalar_t>(),
          relation_key_weight.data_ptr<scalar_t>(),
          relation_value_weight.data_ptr<scalar_t>(),
          relation_value_bias.data_ptr<scalar_t>(),
          sorted_src.data_ptr<int64_t>(),
          sorted_dst.data_ptr<int64_t>(),
          lse.data_ptr<float>(),
          group_dot.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          relation_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed,
          has_value_bias);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      grad_direct_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          r.data_ptr<scalar_t>(),
          relation_key_weight.data_ptr<scalar_t>(),
          relation_value_weight.data_ptr<scalar_t>(),
          relation_value_bias.data_ptr<scalar_t>(),
          sorted_src.data_ptr<int64_t>(),
          sorted_dst.data_ptr<int64_t>(),
          lse.data_ptr<float>(),
          group_dot.data_ptr<float>(),
          grad_q.data_ptr<float>(),
          grad_k.data_ptr<float>(),
          grad_v.data_ptr<float>(),
          grad_r.data_ptr<float>(),
          grad_relation_key_weight.data_ptr<float>(),
          grad_relation_value_weight.data_ptr<float>(),
          grad_relation_value_bias.data_ptr<float>(),
          num_edges,
          num_heads,
          head_dim,
          relation_dim,
          static_cast<float>(scale),
          static_cast<float>(dropout_p),
          seed,
          has_value_bias);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return {
      grad_q,
      grad_k,
      grad_v,
      grad_r,
      grad_relation_key_weight,
      grad_relation_value_weight,
      grad_relation_value_bias,
  };
}
