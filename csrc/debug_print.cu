#include <cstdio>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define AT_DISPATCH_CASE_FLOATING_AND_REDUCED_FLOATING_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_AND_REDUCED_FLOATING_TYPES(__VA_ARGS__))

template <typename float_t>
__global__ void PrintFloatTensor1D(
  float_t* __restrict__ x,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%.4f, ", float(x[i * stride_0]));
  }
  printf("\n");
}

template <typename int_t>
__global__ void PrintIntTensor1D(
  int_t* __restrict__ x,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%lld, ", (int64_t)x[i * stride_0]);
  }
  printf("\n");
}

template <typename float_t>
__global__ void PrintFloatTensor2D(
  float_t* __restrict__ x,
  const size_t shape_0,
  const size_t stride_1,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%.4f, ", float(x[(i / shape_0) * stride_1 + (i % shape_0) * stride_0]));
  }
  printf("\n");
}

template <typename int_t>
__global__ void PrintIntTensor2D(
  int_t* __restrict__ x,
  const size_t shape_0,
  const size_t stride_1,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%lld, ", x[(i / shape_0) * stride_1 + (i % shape_0) * stride_0]);
  }
  printf("\n");
}

template <typename float_t>
__global__ void PrintFloatTensor3D(
  float_t* __restrict__ x,
  const size_t shape_1,
  const size_t shape_0,
  const size_t stride_2,
  const size_t stride_1,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%.4f, ", float(x[(i / shape_0 / shape_1) * stride_2 + ((i / shape_0) % shape_1) * stride_1 + (i % shape_0) * stride_0]));
  }
  printf("\n");
}

template <typename int_t>
__global__ void PrintIntTensor3D(
  int_t* __restrict__ x,
  const size_t shape_1,
  const size_t shape_0,
  const size_t stride_2,
  const size_t stride_1,
  const size_t stride_0,
  const size_t n
) {
  for (size_t i = 0; i < n; ++i) {
    printf("%lld, ", x[(i / shape_0 / shape_1) * stride_2 + ((i / shape_0) % shape_1) * stride_1 + (i % shape_0) * stride_0]);
  }
  printf("\n");
}

void PrintTensor(torch::Tensor x) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  TORCH_CHECK(x.is_cuda(), "The input tensor should be a CUDA tensor");
    if (x.is_floating_point()) {
      if (x.dim() == 1) {
        AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(x.scalar_type(), "PrintFloatTensor1D", ([&] {
          PrintFloatTensor1D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.stride(0), x.numel());
        }));
      } else if (x.dim() == 2) {
        AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(x.scalar_type(), "PrintFloatTensor2D", ([&] {
          PrintFloatTensor2D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.size(1), x.stride(0), x.stride(1), x.numel());
        }));
      } else if (x.dim() == 3) {
        AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(x.scalar_type(), "PrintFloatTensor3D", ([&] {
          PrintFloatTensor3D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.size(1), x.size(2), x.stride(0), x.stride(1), x.stride(2), x.numel());
        }));
      } else {
        // NOTE(Zihao): I'm just too lazy to do this, codegen for higher dimensions should be a better idea
        TORCH_CHECK(false, "Input dimension not supported.");
      }
      cudaError_t status = cudaGetLastError();
      TORCH_CHECK(status == cudaSuccess, "PrintFloatTensor failed with error " + std::string(cudaGetErrorString(status)));
    } else {
      if (x.dim() == 1) {
        AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintIntTensor1D", ([&] {
          PrintIntTensor1D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.stride(0), x.numel());
        }));
      } else if (x.dim() == 2) {
        AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintIntTensor2D", ([&] {
          PrintIntTensor2D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.size(1), x.stride(0), x.stride(1), x.numel());
        }));
      } else if (x.dim() == 3) {
        AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintIntTensor3D", ([&] {
          PrintIntTensor3D<<<1, 1, 0, stream>>>(x.data_ptr<scalar_t>(), x.size(1), x.size(2), x.stride(0), x.stride(1), x.stride(2), x.numel());
        }));
      } else {
        // NOTE(Zihao): I'm just too lazy to do this, codegen for higher dimensions should be a better idea
        TORCH_CHECK(false, "Input dimension not supported.");
      }
      cudaError_t status = cudaGetLastError();
      TORCH_CHECK(status == cudaSuccess, "PrintIntTensor failed with error " + std::string(cudaGetErrorString(status)));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_tensor", &PrintTensor,
        "Print tensor inside cuda kenrels for debugging CUDAGraph");
}
