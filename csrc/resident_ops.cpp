#include <torch/extension.h>

torch::Tensor flat_visible_cuda(torch::Tensor bounds, torch::Tensor planes);
void update_bounds_cuda(torch::Tensor bounds, torch::Tensor ids, torch::Tensor raw);
torch::Tensor paged_visible_cuda(torch::Tensor state, torch::Tensor counts, torch::Tensor requested,
                                 torch::Tensor sky, int64_t block_rows, torch::Tensor planes);
void recompute_page_bounds_cuda(torch::Tensor state, torch::Tensor counts, torch::Tensor mapping,
                                torch::Tensor requested, int64_t block_rows, torch::Tensor bounds);

torch::Tensor gather_parameters_cuda(torch::Tensor state, torch::Tensor slots);
void indexed_adam_cuda(torch::Tensor state, torch::Tensor slots, torch::Tensor raw,
                       torch::Tensor grad, torch::Tensor rates, torch::Tensor frozen,
                       double correction1, double correction2_sqrt);
torch::Tensor upper_cut_cuda(torch::Tensor nodes, torch::Tensor xyz,
                            torch::Tensor bounds, torch::Tensor minimum,
                            torch::Tensor planes, torch::Tensor center,
                            double multiplier, bool culling);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flat_visible", &flat_visible_cuda);
    m.def("update_bounds", &update_bounds_cuda);
    m.def("paged_visible", &paged_visible_cuda);
    m.def("recompute_page_bounds", &recompute_page_bounds_cuda);
    m.def("gather_parameters", &gather_parameters_cuda);
    m.def("indexed_adam", &indexed_adam_cuda);
    m.def("upper_cut", &upper_cut_cuda);
}
