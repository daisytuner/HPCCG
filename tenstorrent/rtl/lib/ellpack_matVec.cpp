#include "ellpack_matVec.hpp"
#include "tt_device_holder.hpp"

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

#ifdef ENABLE_DAISY_RTL
#include <daisy_rtl/daisy_rtl.h>
#endif

#define PAGE_SIZE 512u

namespace tt::daisy {

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

template<typename T>
void pad_buffer(std::vector<T>& out, const T* in, int rows, int cols, int ellpack_cols, T pad_value) {
    // ellpack_cols is the width of the input matrix (in elements)
    // We assume the input is (rows x ellpack_cols)
    
    int num_tiles_r = (rows + 31) / 32;
    int num_tiles_c = (ellpack_cols + 31) / 32;
    
    out.assign(num_tiles_r * num_tiles_c * 32 * 32, pad_value);
    
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            
            auto value = in[row * cols + col];
            out[row * ellpack_cols + col] = value;
        }
    }
}

void print_ellpack(int rows, int cols, const float* vals, const int* addrs) {
    printf("ellpack mat %u x %u :\n", rows, cols);
    
    for (int row = 0; row < rows; row++) {
        printf("  %u: ", row);
        for (int col = 0; col < cols; col++) {
            
            auto addr = addrs[row * cols + col];
            auto value = vals[row * cols + col];
            
            if (addr != UINT32_MAX) {
                printf("%4u:%8.5f ", addr, value);
            } else {
                break; // line done
            }
        }
        printf("\n");
    }
}

#define TT_DEBUG 2

void tt_launch_ellpack_matVecOp(
    tt::tt_metal::IDevice* device,
    uint32_t cells,
    uint32_t ellpack_cols,
    tt::tt_metal::Buffer& d_ellpack_vals,
    tt::tt_metal::Buffer& d_ellpack_addrs,
    tt::tt_metal::Buffer& d_inVec,
    tt::tt_metal::Buffer& d_resVec,
    uint32_t* ellpack_first_col_per_tile,
    uint32_t* ellpack_last_col_per_tile,
    const std::filesystem::path& kernel_dir,
    EllpackHwImpl hwImpl,
    size_t region_id = 0
) {
    // static int invocation = 0;

    // std::cout << "Launching Ellpack MatVec Op (invocation " << invocation++ << ") with hwImpl " << static_cast<int>(hwImpl) << std::endl;

    const bool diag_wb = hwImpl == EllpackHwImpl::FPU;

    tt::tt_metal::Program program;
    // assume 1 tile wide ellpack (in allocation)
    auto ell_used_cols = ellpack_cols;
    auto ell_tiles_total = (cells + 31u) / 32u;
    auto data_format = tt::DataFormat::Float32;
    uint32_t ell_tile_page_size = tt_metal::detail::TileSize(data_format);
    uint32_t vecs_per_page = d_inVec.page_size()/sizeof(float);
    auto vec_page2tile_shift = 2u;
    auto vec2page_shift = static_cast<uint32_t>(std::log2(vecs_per_page));
    auto vecs_per_chunk = 1024u;
    auto vec_chunk2page_shift = 3u;
    auto vec_chunk_size = vecs_per_chunk * sizeof(float);

    uint32_t vector_page_size = vecs_per_page*sizeof(float);
    uint32_t vec_per_tile = 32u;

    uint32_t max_tile_batch_size = diag_wb? 4u : 8u;

    auto avail_cores = device->compute_with_storage_grid_size();

    auto [num_cores, used_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        tt::tt_metal::split_work_to_cores(avail_cores, ell_tiles_total);

    #ifdef ENABLE_DAISY_RTL
        if (region_id != 0) {
            auto num_avail_cores = static_cast<double>(avail_cores.x * avail_cores.y);
            auto num_cores_d = static_cast<double>(num_cores);
            __daisy_instrumentation_metric(region_id, "tt_used_cores", num_cores_d);
            __daisy_instrumentation_metric(region_id, "tt_cores_used_rel", num_cores_d / num_avail_cores);
            __daisy_instrumentation_metric(region_id, "tt_work_units_per_core", ell_tiles_total / num_avail_cores);
        }
    #endif

    #if TT_DEBUG > 0
    std::cout << "Using " << num_cores << " cores to process " << ell_tiles_total << " tiles, " << max_tile_batch_size << " tiles max. per batch ("
              << work_per_core1 << " on " << core_group_1.num_cores() << ", " << work_per_core2 << " on " << core_group_2.num_cores() << "; )" << std::endl;
    #endif

    auto input_tile_count = max_tile_batch_size * 2;
    auto result_page_count = 4;

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

    auto res_buf_page_size = (diag_wb? tt_metal::detail::TileSize(data_format) : (vec_per_tile * sizeof(float)));
    tt_metal::CreateCircularBuffer(
        program,
        used_cores,  // create on all cores
        tt_metal::CircularBufferConfig(
            res_buf_page_size * max_tile_batch_size * 2,
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
            vec_chunk_size * 2,
            {
                {CBIndex::c_3, data_format}
            }
        )
        .set_page_size(CBIndex::c_3, vec_chunk_size)
    );

    std::vector<uint32_t> rd_compile_args, rd_common_args;
    tt_metal::TensorAccessorArgs(d_ellpack_vals).append_to(rd_compile_args, rd_common_args);
    tt_metal::TensorAccessorArgs(d_ellpack_addrs).append_to(rd_compile_args, rd_common_args);
    tt_metal::TensorAccessorArgs(d_inVec).append_to(rd_compile_args, rd_common_args);
    auto kernel_rd_0 = tt_metal::CreateKernel(
        program,
        kernel_dir / "ellpack" / "mat_vec_reader_naive.cpp",
        used_cores,
        tt_metal::ReaderDataMovementConfig(
            rd_compile_args
        )
    );

    std::vector<uint32_t> wr_compile_args, wr_common_args;
    wr_compile_args.push_back(diag_wb? 1u : 0u); // unpack_diag
    tt_metal::TensorAccessorArgs(d_resVec).append_to(wr_compile_args, wr_common_args);
    auto kernel_wr_0 = tt_metal::CreateKernel(
        program,
        kernel_dir / "ellpack" / "vec32_result_wb.cpp", // depends on compile_time arg if it does diag's job
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
        kernel_dir / "ellpack" / (diag_wb? "mat_vec_compute_matmul.cpp" : "mat_vec_compute_naive.cpp"),
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
            d_ellpack_vals.address(),
            d_ellpack_addrs.address(),
            d_inVec.address(),
            max_tile_batch_size,
            vecs_per_page,
            vec2page_shift,
            vec_chunk2page_shift
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
            max_tile_batch_size,
            vecs_per_chunk,
        }
    );

    wr_common_args.insert(
        wr_common_args.begin(),
        {
            d_resVec.address(),
            vecs_per_page,
            vec_page2tile_shift
        }
    );

    tt_metal::SetCommonRuntimeArgs(
        program,
        kernel_wr_0,
        wr_common_args
    );


    uint32_t start_tile = 0;
    uint32_t end_tile = ell_tiles_total; // ex

    for (auto& range : used_cores.ranges()) {

        for (auto& core : range) {
            uint32_t tiles;
            if (core_group_1.contains(core)) {
                tiles = work_per_core1;
            } else if (core_group_2.contains(core)) {
                tiles = work_per_core2;
            } else {
                tiles = 0;
            }

            if (start_tile + tiles > end_tile) {
                tiles = end_tile - start_tile;
            }

            auto first_vec = ellpack_first_col_per_tile[start_tile];
            auto last_vec  = ellpack_last_col_per_tile[start_tile + tiles -1];

            #if TT_DEBUG >= 2
            std::cout << " Core " << core.str() << ": tiles " << start_tile << "..+" << tiles << ", vec " << first_vec << ".." << last_vec << std::endl;
            #endif

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_rd_0,
                core,
                {
                    start_tile,
                    tiles,
                    first_vec,
                    last_vec
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_comp_0,
                core,
                {
                    tiles,
                    first_vec,
                    last_vec
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                kernel_wr_0,
                core,
                {
                    start_tile,
                    tiles,
                }
            );

            start_tile += tiles;
        }
    }

    tt_metal::EnqueueProgram(device->command_queue(0), program, false);
}

void tt_spmv_ellpack(const ELLPACKMatVecParams& params, const float * x, float * y)
{
    _ZN2tt5daisy23tt_ellpack_matVec(
        params.vals,
        params.inds,
        params.nrow,
        params.ncol,
        params.ellpack_nnz,
        params.ellpack_cols,
        params.row_min_cols,
        params.row_max_cols,
        x,
        y
    );
}

}   // namespace tt::daisy

void _ZN2tt5daisy23tt_ellpack_matVec(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    tt::tt_metal::IDevice* device = tt::daisy::get_device();

    // Copy-in vals
    void * d_ellpack_vals = _ZN2tt5daisy23tt_ellpack_matVec_in_0(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    );

    // Copy-in inds
    void * d_ellpack_addrs = _ZN2tt5daisy23tt_ellpack_matVec_in_1(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    );

    // Copy-in x
    void * d_inVec = _ZN2tt5daisy23tt_ellpack_matVec_in_8(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    );

    // Allocate output y
    void * d_resVec = _ZN2tt5daisy23tt_ellpack_matVec_in_9(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    );

    // Launch kernel
    _ZN2tt5daisy23tt_ellpack_matVec_kernel(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y,
        d_ellpack_vals,
        d_ellpack_addrs,
        d_inVec,
        d_resVec
    );

    // Copy-out y
    _ZN2tt5daisy23tt_ellpack_matVec_out_9(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y,
        d_resVec
    );

    delete static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_ellpack_vals);
    delete static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_ellpack_addrs);
    delete static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_inVec);
    delete static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_resVec);
}

std::shared_ptr<tt::tt_metal::Buffer> _ZN2tt5daisy23tt_ellpack_matVec_in_0_impl(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
#ifdef ENABLE_DAISY_RTL
    __daisy_metadata_t metadata_in = {
        .file_name = "ellpack_matVec.cpp",
        .function_name = "tt::daisy::ellpack_matVec",
        .line_begin = 506,
        .line_end = 579,
        .column_begin = 0,
        .column_end = 0,
        .element_type = "h2d_transfer",
        .target_type = "TENSTORRENT",
        .region_uuid = "foam_ellpack_matVec_in0"
    };
    unsigned long long region = __daisy_instrumentation_init(&metadata_in, __DAISY_EVENT_SET_NONE);
    __daisy_instrumentation_enter(region);
#endif

    tt::tt_metal::IDevice* device = tt::daisy::get_device();
    
    std::vector<float> tilized_vals;
    tt::daisy::tilize_buffer(tilized_vals, vals, nrow, ellpack_cols, ellpack_cols, 0.0f);
    
    size_t vals_buffer_size = tilized_vals.size() * sizeof(float);
    size_t vals_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);
    
    std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_vals;
    size_t aligned_vals_size = ((vals_buffer_size + vals_tile_size - 1) / vals_tile_size) * vals_tile_size;
    d_ellpack_vals = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
        .device = device,
        .size = aligned_vals_size,
        .page_size = vals_buffer_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    });

    tt::tt_metal::EnqueueWriteBuffer(
        device->command_queue(0),
        d_ellpack_vals,
        tilized_vals.data(),
    #ifdef ENABLE_DAISY_RTL
        true
    #else 
        false
    #endif
    );

#ifdef ENABLE_DAISY_RTL
    __daisy_instrumentation_exit(region);
    __daisy_instrumentation_increment(region, "pcie_bytes", vals_buffer_size);
    __daisy_instrumentation_finalize(region);
#endif

  return d_ellpack_vals;
}

void * _ZN2tt5daisy23tt_ellpack_matVec_in_0(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    return new std::shared_ptr<tt::tt_metal::Buffer>(_ZN2tt5daisy23tt_ellpack_matVec_in_0_impl(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    ));
}

std::shared_ptr<tt::tt_metal::Buffer> _ZN2tt5daisy23tt_ellpack_matVec_in_1_impl(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
#ifdef ENABLE_DAISY_RTL
    __daisy_metadata_t metadata_in = {
        .file_name = "ellpack_matVec.cpp",
        .function_name = "tt::daisy::ellpack_matVec",
        .line_begin = 608,
        .line_end = 681,
        .column_begin = 0,
        .column_end = 0,
        .element_type = "h2d_transfer",
        .target_type = "TENSTORRENT",
        .region_uuid = "foam_ellpack_matVec_in1"
    };
    unsigned long long region = __daisy_instrumentation_init(&metadata_in, __DAISY_EVENT_SET_NONE);
    __daisy_instrumentation_enter(region);
#endif

    tt::tt_metal::IDevice* device = tt::daisy::get_device();
    
    std::vector<uint32_t> padded_inds;
    tt::daisy::pad_buffer(padded_inds, (const uint32_t*)inds, nrow, ellpack_cols, ellpack_cols, (uint32_t)UINT32_MAX);

    tt::daisy::print_ellpack(nrow, ellpack_cols, vals, inds);
    
    size_t addrs_buffer_size = padded_inds.size() * sizeof(uint32_t);
    size_t addrs_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::UInt32);
  
    std::shared_ptr<tt::tt_metal::Buffer> d_ellpack_addrs;
    size_t aligned_addrs_size = ((addrs_buffer_size + addrs_tile_size - 1) / addrs_tile_size) * addrs_tile_size;
    d_ellpack_addrs = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = aligned_addrs_size,
      .page_size = addrs_tile_size,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });

    tt::tt_metal::EnqueueWriteBuffer(
        device->command_queue(0),
        d_ellpack_addrs,
        padded_inds.data(),
    #ifdef ENABLE_DAISY_RTL
        true
    #else 
        false
    #endif
  );

#ifdef ENABLE_DAISY_RTL
    __daisy_instrumentation_exit(region);
    __daisy_instrumentation_increment(region, "pcie_bytes", addrs_buffer_size);
    __daisy_instrumentation_finalize(region);
#endif

  return d_ellpack_addrs;
}

void * _ZN2tt5daisy23tt_ellpack_matVec_in_1(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    return new std::shared_ptr<tt::tt_metal::Buffer>(_ZN2tt5daisy23tt_ellpack_matVec_in_1_impl(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    ));
}

std::shared_ptr<tt::tt_metal::Buffer> _ZN2tt5daisy23tt_ellpack_matVec_in_8_impl(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
#ifdef ENABLE_DAISY_RTL
    __daisy_metadata_t metadata_in = {
        .file_name = "ellpack_matVec.cpp",
        .function_name = "tt::daisy::ellpack_matVec",
        .line_begin = 710,
        .line_end = 781,
        .column_begin = 0,
        .column_end = 0,
        .element_type = "h2d_transfer",
        .target_type = "TENSTORRENT",
        .region_uuid = "foam_ellpack_matVec_in8"
    };
    unsigned long long region = __daisy_instrumentation_init(&metadata_in, __DAISY_EVENT_SET_NONE);
    __daisy_instrumentation_enter(region);
#endif

    tt::tt_metal::IDevice* device = tt::daisy::get_device();
    
    // Create buffers for input and output vectors
    size_t inVec_buffer_size = sizeof(float) * ncol;

    printf("vector (%u) :\n", nrow);
    
    for (int row = 0; row < nrow; row++) {
        auto value = vals[row];
        printf("  %u: %8.5f\n", row, value);
    }
  
    std::shared_ptr<tt::tt_metal::Buffer> d_inVec;
    // Align buffer size to be divisible by page size for interleaved buffers
    size_t aligned_inVec_size = ((inVec_buffer_size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    d_inVec = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
      .device = device,
      .size = aligned_inVec_size,
      .page_size = PAGE_SIZE,
      .buffer_type = tt::tt_metal::BufferType::DRAM
    });

    tt::tt_metal::EnqueueWriteBuffer(
        device->command_queue(0),
        d_inVec,
        x,
    #ifdef ENABLE_DAISY_RTL
        true
    #else 
        false
    #endif
    );

#ifdef ENABLE_DAISY_RTL
    __daisy_instrumentation_exit(region);
    __daisy_instrumentation_increment(region, "pcie_bytes", inVec_buffer_size);
    __daisy_instrumentation_finalize(region);
#endif

  return d_inVec;
}

void * _ZN2tt5daisy23tt_ellpack_matVec_in_8(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    return new std::shared_ptr<tt::tt_metal::Buffer>(_ZN2tt5daisy23tt_ellpack_matVec_in_8_impl(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    ));
}

std::shared_ptr<tt::tt_metal::Buffer> _ZN2tt5daisy23tt_ellpack_matVec_in_9_impl(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    tt::tt_metal::IDevice* device = tt::daisy::get_device();
    
    size_t resVec_buffer_size = sizeof(float) * nrow;
    
    std::shared_ptr<tt::tt_metal::Buffer> d_resVec;
    // Align buffer size to be divisible by page size for interleaved buffers
    size_t aligned_resVec_size = ((resVec_buffer_size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    d_resVec = tt::tt_metal::CreateBuffer(tt::tt_metal::BufferConfig{
        .device = device,
        .size = aligned_resVec_size,
        .page_size = PAGE_SIZE,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    });

    return d_resVec;
}

void * _ZN2tt5daisy23tt_ellpack_matVec_in_9(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y
)
{
    return new std::shared_ptr<tt::tt_metal::Buffer>(_ZN2tt5daisy23tt_ellpack_matVec_in_9_impl(
        vals,
        inds,
        nrow,
        ncol,
        ellpack_nnz,
        ellpack_cols,
        row_min_cols,
        row_max_cols,
        x,
        y
    ));
}

void _ZN2tt5daisy23tt_ellpack_matVec_kernel(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y,
    void * d_ellpack_vals_ptr,
    void * d_ellpack_addrs_ptr,
    void * d_inVec_ptr,
    void * d_resVec_ptr
)
{
#ifdef ENABLE_DAISY_RTL
    __daisy_metadata_t metadata_kernel = {
        .file_name = "ellpack_matVec.cpp",
        .function_name = "tt::daisy::ellpack_matVec",
        .line_begin = 876,
        .line_end = 958,
        .column_begin = 0,
        .column_end = 0,
        .element_type = "map",
        .target_type = "TENSTORRENT",
        .region_uuid = "foam_ellpack_matVec_kernel"
    };
    unsigned long long region = __daisy_instrumentation_init(&metadata_kernel, __DAISY_EVENT_SET_NONE);
    __daisy_instrumentation_enter(region);
#else
    unsigned long long region = 0;
#endif

    auto d_ellpack_vals = *static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_ellpack_vals_ptr);
    auto d_ellpack_addrs = *static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_ellpack_addrs_ptr);
    auto d_inVec = *static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_inVec_ptr);
    auto d_resVec = *static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_resVec_ptr);

    tt::tt_metal::IDevice* device = tt::daisy::get_device();
  
    int num_tiles_r = (nrow + 31) / 32;
    std::vector<uint32_t> first_cols(num_tiles_r);
    std::vector<uint32_t> last_cols(num_tiles_r);
    for (int tr = 0; tr < num_tiles_r; ++tr) {
        uint32_t min_val = UINT32_MAX;
        uint32_t max_val = 0;
        for (int r = 0; r < 32; ++r) {
            int global_r = tr * 32 + r;
            if (global_r >= nrow) break;
            uint32_t row_min = row_min_cols[global_r];
            uint32_t row_max = row_max_cols[global_r];
            if (row_min < min_val) min_val = row_min;
            if (row_max > max_val) max_val = row_max;
        }
        first_cols[tr] = min_val;
        last_cols[tr] = max_val;
    }

    // Launch the matrix-vector multiplication kernel
    const char* e = std::getenv("TT_HPCCG_KERNEL_DIR");
    if (!e) {
        e = "tenstorrent/rtl/kernels";
    }
    std::filesystem::path kernel_dir = std::filesystem::path(e);
    tt::daisy::tt_launch_ellpack_matVecOp(
        device,
        nrow,
        ellpack_cols,
        *d_ellpack_vals,
        *d_ellpack_addrs,
        *d_inVec,
        *d_resVec,
        first_cols.data(),
        last_cols.data(),
        kernel_dir,
        tt::daisy::EllpackHwImpl::None,
        region
    );

#ifdef ENABLE_DAISY_RTL
    tt::tt_metal::Finish(device->command_queue(0)); 

    __daisy_instrumentation_exit(region);
    __daisy_instrumentation_increment(region, "flop", ellpack_nnz * 2);
    __daisy_instrumentation_increment(region, "dram_bytes", ellpack_nnz * (sizeof(float) + sizeof(uint32_t)) + ncol * sizeof(float) + nrow * sizeof(float));
    __daisy_instrumentation_finalize(region);
#endif
}

void _ZN2tt5daisy23tt_ellpack_matVec_out_9(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y,
    void * d_resVec_ptr
)
{
#ifdef ENABLE_DAISY_RTL
    __daisy_metadata_t metadata_in = {
        .file_name = "ellpack_matVec.cpp",
        .function_name = "tt::daisy::ellpack_matVec",
        .line_begin = 960,
        .line_end = 1009,
        .column_begin = 0,
        .column_end = 0,
        .element_type = "d2h_transfer",
        .target_type = "TENSTORRENT",
        .region_uuid = "foam_ellpack_matVec_out9"
    };
    unsigned long long region = __daisy_instrumentation_init(&metadata_in, __DAISY_EVENT_SET_NONE);
    __daisy_instrumentation_enter(region);
#endif

    auto d_resVec = *static_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(d_resVec_ptr);
    tt::tt_metal::IDevice* device = tt::daisy::get_device();

    tt::tt_metal::EnqueueReadBuffer(
        device->command_queue(0),
        d_resVec,
        y,
        false
    );

    tt::tt_metal::Finish(device->command_queue(0)); 

#ifdef ENABLE_DAISY_RTL
    size_t resVec_buffer_size = sizeof(float) * nrow;

    __daisy_instrumentation_exit(region);
    __daisy_instrumentation_increment(region, "pcie_bytes", resVec_buffer_size);
    __daisy_instrumentation_finalize(region);
#endif
}

extern "C" void _ZN2tt5daisy23tt_ellpack_matVec_out_0(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y,
    void * d_ellpack_vals
) {};

extern "C" void _ZN2tt5daisy23tt_ellpack_matVec_out_1(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y,
    void * d_ellpack_addrs
) {};

extern "C" void _ZN2tt5daisy23tt_ellpack_matVec_out_8(
    const float * vals,
    const int * inds,
    int nrow,
    int ncol,
    int ellpack_nnz,
    int ellpack_cols,
    const uint32_t* row_min_cols,
    const uint32_t* row_max_cols,
    const float * x,
    float * y,
    void * d_inVec
) {};
