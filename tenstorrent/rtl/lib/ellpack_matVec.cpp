#include "ellpack_matVec.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"

#include <cstdint>
#include <cstdlib>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <filesystem>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::daisy {

// Global device pointer
static tt_metal::IDevice* g_device = nullptr;

// Initialize the global device
static tt_metal::IDevice* get_device() {
    if (g_device == nullptr) {
        g_device = tt_metal::CreateDevice(0);
        // Register cleanup function to run at program exit
        std::atexit([]() {
            if (g_device != nullptr) {
                tt_metal::CloseDevice(g_device);
                g_device = nullptr;
            }
        });
    }
    return g_device;
}

template<typename T>
void tilize_buffer(std::vector<T>& out, const T* in, int rows, int cols, int ellpack_cols, T pad_value) {
    // ellpack_cols is the width of the input matrix (in elements)
    // We assume the input is (rows x ellpack_cols)
    
    int num_tiles_r = (rows + 31) / 32;
    int num_tiles_c = (ellpack_cols + 31) / 32;
    
    out.assign(num_tiles_r * num_tiles_c * 32 * 32, pad_value);
    
    for (int tr = 0; tr < num_tiles_r; tr++) {
        for (int tc = 0; tc < num_tiles_c; tc++) {
            T* tile_start = &out[(tr * num_tiles_c + tc) * 1024];
            
            // Face 0
            for (int r = 0; r < 16; r++) {
                for (int c = 0; c < 16; c++) {
                    int global_r = tr * 32 + r;
                    int global_c = tc * 32 + c;
                    if (global_r < rows && global_c < ellpack_cols) {
                        tile_start[r * 16 + c] = in[global_r * ellpack_cols + global_c];
                    }
                }
            }
            
            // Face 1
            for (int r = 0; r < 16; r++) {
                for (int c = 0; c < 16; c++) {
                    int global_r = tr * 32 + r;
                    int global_c = tc * 32 + 16 + c;
                    if (global_r < rows && global_c < ellpack_cols) {
                        tile_start[256 + r * 16 + c] = in[global_r * ellpack_cols + global_c];
                    }
                }
            }
            
            // Face 2
            for (int r = 0; r < 16; r++) {
                for (int c = 0; c < 16; c++) {
                    int global_r = tr * 32 + 16 + r;
                    int global_c = tc * 32 + c;
                    if (global_r < rows && global_c < ellpack_cols) {
                        tile_start[512 + r * 16 + c] = in[global_r * ellpack_cols + global_c];
                    }
                }
            }
            
            // Face 3
            for (int r = 0; r < 16; r++) {
                for (int c = 0; c < 16; c++) {
                    int global_r = tr * 32 + 16 + r;
                    int global_c = tc * 32 + 16 + c;
                    if (global_r < rows && global_c < ellpack_cols) {
                        tile_start[768 + r * 16 + c] = in[global_r * ellpack_cols + global_c];
                    }
                }
            }
        }
    }
}

#define TT_DEBUG 0

void tt_launch_ellpack_matVecOp(
    tt::tt_metal::IDevice* device,
    uint32_t cell_count,
    uint32_t ellpack_cols,
    std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_vals,
    std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_addrs,
    tt::tt_metal::Buffer& d_inVec,
    tt::tt_metal::Buffer& d_resVec,
    const std::filesystem::path& kernel_dir,
    EllpackHwImpl hwImpl
) {
    // static int invocation = 0;

    // std::cout << "Launching Ellpack MatVec Op (invocation " << invocation++ << ") with hwImpl " << static_cast<int>(hwImpl) << std::endl;

    const bool diag_wb = hwImpl == EllpackHwImpl::FPU;

    tt::tt_metal::Program program;
    // assume 1 tile wide ellpack (in allocation)
    auto ell_used_cols = ellpack_cols;
    auto ell_tiles_total = (cell_count + 31u) / 32u;
    auto data_format = tt::DataFormat::Float32;
    uint32_t ell_tile_page_size = tt_metal::detail::TileSize(data_format);
    int vector_page_size = 1024;
    auto vec_entries_per_chunk = vector_page_size / 4u;
    auto vec_tile_h_per_chunk = vec_entries_per_chunk / 32u;
    auto vec_chunks_total = (ell_tiles_total + vec_tile_h_per_chunk -1) / vec_tile_h_per_chunk; // 1024 byte pages -> 256 floats -> 8 tiles for each vec-chunk

    uint32_t batch_size = diag_wb? 4u : 8u;

    auto batches_total = (ell_tiles_total + batch_size - 1) / batch_size;

    auto avail_cores = device->compute_with_storage_grid_size();

    auto [num_cores, used_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        tt::tt_metal::split_work_to_cores(avail_cores, batches_total);

    #if TT_DEBUG > 0
    std::cout << "Using " << num_cores << " cores to process " << batches_total << " batches, " << batch_size << " tiles each ("
              << work_per_core1 << " on " << core_group_1.num_cores() << ", " << work_per_core2 << " on " << core_group_2.num_cores() << "; " << vec_chunks_total << " vec chunks (" << vec_entries_per_chunk << " floats/chunk))" << std::endl;
    #endif

    auto input_tile_count = batch_size * 2;
    auto vector_chunk_count = 32u; // at least
    auto result_page_count = 4;

    size_t vector_size = vector_page_size * 2;

    // c0 output (vector)
    // c1 input (mat - ellpack data)
    // c2 input (mat - ellpack addr)
    // c3 input (vector)
    // c4 temp (mat - mulmat of vector matched to ellpack tiles)
    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            ell_tile_page_size * input_tile_count,
            {
                {CBIndex::c_1, data_format},
            }
        )
        .set_page_size(CBIndex::c_1, ell_tile_page_size)
    );

    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            ell_tile_page_size * input_tile_count,
            {
                {CBIndex::c_4, data_format},
            }
        )
        .set_page_size(CBIndex::c_4, ell_tile_page_size)
    );

    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            ell_tile_page_size * input_tile_count,
            {
                {CBIndex::c_2, tt::DataFormat::Float32} // we MUST lie, because only then will tt-runtime unlock TF32 support (the host runtime maps the "dest" type, which is used for DST and SrcA/SrcB regs, Tf32 cannot be manually set, but must be chosed as Dest type to not loose precision)
                // tensix driver will use the dest type as is for unpack byte size. And will use it as is for SrcA/SrcB input type (which if FP32 will expect FP16 and therefore misinterpret the unpacked data)
                // CBs throw, if we try to use them with TF32 explicitly, because there is a switch case that does not define a byte-size
                // default mapping will either map FP16 as dest type (by default) or FP32 (if UnpackToDestFp32 is set. No other effects on the host side)
            }
        )
        .set_page_size(CBIndex::c_2, ell_tile_page_size));

    auto res_buf_page_size = (diag_wb? tt_metal::detail::TileSize(data_format) : vector_page_size);
    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            res_buf_page_size * result_page_count,
            {
                {CBIndex::c_0, data_format},
            }
        )
        .set_page_size(CBIndex::c_0, res_buf_page_size)
    );

    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            vector_page_size * vector_chunk_count,
            {
                {CBIndex::c_3, data_format}
            }
        )
        .set_page_size(CBIndex::c_3, vector_page_size)
    );

    std::vector<uint32_t> rd_compile_args, rd_common_args;
    tt_metal::TensorAccessorArgs(d_ellpack_vals).append_to(rd_compile_args, rd_common_args);
    tt_metal::TensorAccessorArgs(d_ellpack_addrs).append_to(rd_compile_args, rd_common_args);
    tt_metal::TensorAccessorArgs(d_inVec).append_to(rd_compile_args, rd_common_args);
    auto kernel_rd_0 = tt_metal::CreateKernel(
        program,
        kernel_dir / (0? "mat_vec_reader_collection.cpp" : "mat_vec_reader_naive.cpp"),
        used_cores,
        tt_metal::ReaderDataMovementConfig(
            rd_compile_args
        )
    );

    std::vector<uint32_t> wr_compile_args, wr_common_args;
    tt_metal::TensorAccessorArgs(d_resVec).append_to(wr_compile_args, wr_common_args);
    auto kernel_wr_0 = tt_metal::CreateKernel(
        program,
        kernel_dir / (diag_wb? "vec_diag_result_wb.cpp" : "vec_bare_result_wb.cpp"),
        used_cores,
        tt_metal::WriterDataMovementConfig(
            wr_compile_args
        )
    );

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // unpack_modes[CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    // unpack_modes[CBIndex::c_4] = UnpackToDestMode::UnpackToDestFp32;

    auto kernel_comp_0 = tt_metal::CreateKernel(
        program,
        kernel_dir / (diag_wb? "mat_vec_compute_matmul.cpp" : "mat_vec_compute_naive.cpp"),
        used_cores,
        tt_metal::ComputeConfig {
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            // .dst_full_sync_en = false,
            // .unpack_to_dest_mode = unpack_modes,
            // .math_approx_mode = false,
            .compile_args = {},
        }
    );

    rd_common_args.insert(
        rd_common_args.begin(),
        {
            d_ellpack_vals->address(),
            d_ellpack_addrs->address(),
            d_inVec.address(),
            vec_chunks_total,
            1, // vec_chunk_batch_size
        }
    );

    tt_metal::SetCommonRuntimeArgs(
        program,
        kernel_rd_0,
        rd_common_args
    );

    tt_metal::SetCommonRuntimeArgs(
        program,
        kernel_comp_0,
        {
            vec_chunks_total,
            1, // vec_chunk_batch_size
            1 // stream vec
        }
    );

    wr_common_args.insert(
        wr_common_args.begin(),
        {
            d_resVec.address(),
            batch_size,
        }
    );

    tt_metal::SetCommonRuntimeArgs(
        program,
        kernel_wr_0,
        wr_common_args
    );


    uint32_t start_batch = 0;
    uint32_t end_tile = ell_tiles_total; // ex

    for (auto& range : used_cores.ranges()) {

        for (auto& core : range) {
            uint32_t units;
            if (core_group_1.contains(core)) {
                units = work_per_core1;
            } else if (core_group_2.contains(core)) {
                units = work_per_core2;
            } else {
                units = 0;
            }

            auto tiles = units * batch_size;
            auto start_tile = start_batch * batch_size;
            if (start_tile + tiles > end_tile) {
                tiles = end_tile - start_tile;
            }

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_rd_0,
                core,
                {
                    start_tile,
                    tiles,
                    units,
                    batch_size
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_comp_0,
                core,
                {
                    units,
                    batch_size,
                    tiles
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_wr_0,
                core,
                {
                    start_batch,
                    units,
                    start_tile,
                    tiles,
                }
            );

            start_batch += units;
        }
    }

    tt_metal::EnqueueProgram(device->command_queue(0), program, false);
}

void tt_ellpack_matVec(int nrow, int ncol, int nnz, int ellpack_cols,
                     const float * const vals,
                     const int * const inds,
		 const float * const x, float * const y)
{
  tt_metal::IDevice* device = get_device();
  
  // Tilize matrix values and indices
  std::vector<float> tilized_vals;
  tilize_buffer(tilized_vals, vals, nrow, ellpack_cols, ellpack_cols, 0.0f);
  
  std::vector<uint32_t> tilized_inds;
  // Cast int* to uint32_t* for tilize_buffer
  tilize_buffer(tilized_inds, (const uint32_t*)inds, nrow, ellpack_cols, ellpack_cols, (uint32_t)UINT32_MAX);

  // Create buffers for matrix values and indices
  size_t vals_buffer_size = tilized_vals.size() * sizeof(float);
  size_t vals_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float32);
  
  std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_vals;
  if (vals_buffer_size > vals_tile_size) {
    // Align buffer size to be divisible by tile size for interleaved buffers
    size_t aligned_vals_size = ((vals_buffer_size + vals_tile_size - 1) / vals_tile_size) * vals_tile_size;
    d_ellpack_vals = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
      .device = device,
      .size = aligned_vals_size,
      .page_size = vals_tile_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  } else {
    d_ellpack_vals = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = vals_buffer_size,
      .page_size = vals_buffer_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  }

  size_t addrs_buffer_size = tilized_inds.size() * sizeof(uint32_t);
  size_t addrs_tile_size = tt_metal::detail::TileSize(tt::DataFormat::UInt32);
  
  std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_addrs;
  if (addrs_buffer_size > addrs_tile_size) {
    // Align buffer size to be divisible by tile size for interleaved buffers
    size_t aligned_addrs_size = ((addrs_buffer_size + addrs_tile_size - 1) / addrs_tile_size) * addrs_tile_size;
    d_ellpack_addrs = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
      .device = device,
      .size = aligned_addrs_size,
      .page_size = addrs_tile_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  } else {
    d_ellpack_addrs = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = addrs_buffer_size,
      .page_size = addrs_buffer_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  }

  // Create buffers for input and output vectors
  size_t inVec_buffer_size = sizeof(float) * ncol;
  size_t page_size = 1024;
  
  std::shared_ptr<tt::tt_metal::Buffer> d_inVec;
  if (inVec_buffer_size > page_size) {
    // Align buffer size to be divisible by page size for interleaved buffers
    size_t aligned_inVec_size = ((inVec_buffer_size + page_size - 1) / page_size) * page_size;
    d_inVec = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
      .device = device,
      .size = aligned_inVec_size,
      .page_size = page_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  } else {
    d_inVec = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = inVec_buffer_size,
      .page_size = inVec_buffer_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  }

  size_t resVec_buffer_size = sizeof(float) * nrow;
  
  std::shared_ptr<tt::tt_metal::Buffer> d_resVec;
  if (resVec_buffer_size > page_size) {
    // Align buffer size to be divisible by page size for interleaved buffers
    size_t aligned_resVec_size = ((resVec_buffer_size + page_size - 1) / page_size) * page_size;
    d_resVec = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
      .device = device,
      .size = aligned_resVec_size,
      .page_size = page_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  } else {
    d_resVec = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = resVec_buffer_size,
      .page_size = resVec_buffer_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });
  }  
  // Copy matrix data to device
  tt::tt_metal::EnqueueWriteBuffer(
    device->command_queue(0),
    d_ellpack_vals,
    tilized_vals.data(),
    true
  );

  tt::tt_metal::EnqueueWriteBuffer(
    device->command_queue(0),
    d_ellpack_addrs,
    tilized_inds.data(),
    false
  );

  // Copy input vector to device
  tt::tt_metal::EnqueueWriteBuffer(
    device->command_queue(0),
    d_inVec,
    x,
    false
  );
  // Launch the matrix-vector multiplication kernel
  tt::daisy::tt_launch_ellpack_matVecOp(
    device,
    nrow,  // cell_count should be number of rows
    ellpack_cols,   // ellpack_cols passed from caller
    d_ellpack_vals,
    d_ellpack_addrs,
    *d_inVec,
    *d_resVec,
    std::filesystem::path("/home/lukas/repos/HPCCG/tenstorrent/rtl/kernels"),  // kernel directory
    tt::daisy::EllpackHwImpl::None  // hardware implementation type
  );
  // Read result back to host with non-blocking call
  tt::tt_metal::EnqueueReadBuffer(
    device->command_queue(0),
    d_resVec,
    y,
    false
  );

  tt::tt_metal::Finish(device->command_queue(0));  
}

}   // namespace tt::daisy
