/**
 * @file thread_mapped.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication kernels.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t rows,
                                const std::size_t cols,
                                const std::size_t nnz,
                                const index_t* row_indices,
                                const index_t* col_indices,
                                const type_t* values,
                                const type_t* x,
                                type_t* y) {
  /// Requires sorted COO.
  auto count_entries = thrust::make_transform_iterator(
      row_indices, [row_indices] __host__ __device__(const index_t& i) {
        auto it = thrust::lower_bound(
            thrust::seq, thrust::counting_iterator<index_t>(0),
            thrust::counting_iterator<index_t>(nnz), i,
            [row_indices] __host__ __device__(const index_t& pivot,
                                              const index_t& key) {
              return row_indices[pivot] < key;
            });

        return (*it);
      });

  /// Equivalent to:
  /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
  /// row < rows; (boundary condition)
  /// row += gridDim.x * blockDim.x. (step)
  for (auto row : config.tiles()) {
    type_t sum = 0;

    /// Equivalent to:
    /// for (offset_t nz = offset; nz < end; ++nz)
    for (auto nz : config.atoms(row)) {
      sum += values[nz] * x[indices[nz]];
    }

    // Output
    y[row] = sum;
  }
}

/**
 * @brief Sparse-Matrix Vector Multiplication API.
 *
 * @tparam index_t Type of indices.
 * @tparam type_t Type of values.
 * @param coo CSR matrix (GPU).
 * @param x Input vector x (GPU).
 * @param y Output vector y (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename type_t>
void thread_mapped(csr_t<index_t, type_t>& coo,
                   vector_t<type_t>& x,
                   vector_t<type_t>& y,
                   cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.

  // Create a schedule.
  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, index_t>;

  row_indices_data = coo.row_indices.data().get();
  num_rows = coo.rows;

  setup_t config(row_indices_data, num_rows, coo.nnzs);

  std::size_t grid_size = (coo.rows + block_size - 1) / block_size;
  launch::non_cooperative(stream, __thread_mapped<setup_t, index_t, type_t>,
                          grid_size, block_size, config, coo.rows, coo.cols,
                          coo.nnzs, coo.row_indices.data().get(),
                          coo.col_indices.data().get(), coo.values.data().get(),
                          x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops