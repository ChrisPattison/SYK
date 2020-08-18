#pragma once
#include <cstdint>

namespace util {
// http://www.netlib.org/scalapack/slug/node78.html
/** Container for holding onto distributed matrices in block cyclic layout
 */
template<typename matrix_type>
struct distributed_matrix {
    matrix_type local_matrix;

    std::size_t size_x;
    std::size_t size_y;

    std::size_t num_procs_x;
    std::size_t num_procs_y;

    std::size_t proc_idx_x;
    std::size_t proc_idx_y;
    
    std::size_t block_size_x;
    std::size_t block_size_y;
};
}
