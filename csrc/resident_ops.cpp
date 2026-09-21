#include <torch/extension.h>

torch::Tensor gather_parameters_cuda(torch::Tensor state, torch::Tensor slots);
void indexed_adam_cuda(torch::Tensor state, torch::Tensor slots, torch::Tensor raw,
                       torch::Tensor grad, torch::Tensor rates, torch::Tensor frozen,
                       double correction1, double correction2_sqrt);
torch::Tensor upper_cut_cuda(torch::Tensor nodes, torch::Tensor xyz,
                            torch::Tensor bounds, torch::Tensor minimum,
                            torch::Tensor planes, torch::Tensor center,
                            double multiplier, bool culling);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_parameters", &gather_parameters_cuda);
    m.def("indexed_adam", &indexed_adam_cuda);
    m.def("upper_cut", &upper_cut_cuda);
}
