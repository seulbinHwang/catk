#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> segmented_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_edge,
    torch::Tensor v_edge,
    torch::Tensor sorted_dst,
    torch::Tensor dst_ptr,
    double scale,
    double dropout_p,
    std::uint64_t seed);

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
    std::uint64_t seed);

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
    bool has_value_bias);

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
    bool has_value_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &segmented_attention_forward_cuda, "CAT-K segmented graph attention forward");
  m.def("backward", &segmented_attention_backward_cuda, "CAT-K segmented graph attention backward");
  m.def("forward_direct", &segmented_attention_forward_direct_cuda, "CAT-K direct relation segmented graph attention forward");
  m.def("backward_direct", &segmented_attention_backward_direct_cuda, "CAT-K direct relation segmented graph attention backward");
}
