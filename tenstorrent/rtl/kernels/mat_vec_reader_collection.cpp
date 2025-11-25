
#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"

#include "debug/dprint.h"

#ifndef FACE_LAYOUT
#define FACE_LAYOUT 1
#endif

static bool collect_for(float* result, uint32_t addr, float* vec_ptr, uint32_t vec_chunk_offset, uint32_t vec_chunk_offset_end, uint32_t rowIdx, uint32_t colIdx) {

    if (addr < vec_chunk_offset || addr >= vec_chunk_offset_end) {
        if (addr == UINT32_MAX) {
            return true;
        }
        // DPRINT << " col [" << rowIdx << ", " << colIdx << "]: " << addr << " not in range" << ENDL();
    } else {
        auto val = vec_ptr[addr - vec_chunk_offset];
        *result = val;

        // DPRINT << " col [" << rowIdx << ", " << colIdx << "] = " << val << " from " << addr << ENDL();
    }

    return false;
}

static inline void collect_for_mul_tile(uint32_t* addr_ptr, float* collect_ptr, float* vec_ptr, uint32_t vec_chunk_offset, uint32_t vecs_per_chunk) {
    const uint32_t vec_chunk_end = vec_chunk_offset + vecs_per_chunk;

    constexpr uint32_t NEXT_ROW_OFFSET = (FACE_LAYOUT == 1) ? 16 : 32;
    constexpr uint32_t NEXT_FACE_ROW_OFFSET = (FACE_LAYOUT == 1) ? (16*16 + 16) : 32;
    constexpr uint32_t NEXT_FACE_COL_OFFSET = (FACE_LAYOUT == 1) ? (16*16 - 15) : 1;


    float* collect_outer = collect_ptr;
    int i = 0;
    for (int rowIdx = 0; rowIdx < 32; ++rowIdx) {
        uint32_t* addr_row = addr_ptr + rowIdx * 32;

        float* collect = collect_outer;
        bool line_done = false;
        for (int colIdx = 0; colIdx < 32; ++colIdx) {

            uint32_t adr = addr_row[colIdx];

            if (!line_done) {
                line_done |= collect_for(collect, adr, vec_ptr, vec_chunk_offset, vec_chunk_end, rowIdx, colIdx);
            }
            if (line_done) {
                *collect = 0.0f;
                // break;
            }
            // *collect = (i++)*1.0f;
            // *collect = 1.0f;

            if (colIdx == 15) {
                // collect += NEXT_FACE_ROW_OFFSET;
                collect += NEXT_FACE_COL_OFFSET;
            } else {
                // collect += NEXT_ROW_OFFSET;
                collect += 1;
            }
        }
        if (rowIdx == 15) {
            // collect_outer += NEXT_FACE_COL_OFFSET;
            collect_outer += NEXT_FACE_ROW_OFFSET;
        } else {
            collect_outer += NEXT_ROW_OFFSET;
        }
    }
}


void kernel_main() {
    uint32_t ell_val_base_addr = get_common_arg_val<uint32_t>(0);
    uint32_t ell_addr_base_addr = get_common_arg_val<uint32_t>(1);
    uint32_t inVec_base_addr = get_common_arg_val<uint32_t>(2);
    uint32_t vec_chunks = get_common_arg_val<uint32_t>(3);
    uint32_t vec_chunk_batch_size = get_common_arg_val<uint32_t>(4);

    uint32_t first_tile_offset = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1); // how many tiles to read from the matrix
    uint32_t batches = get_arg_val<uint32_t>(2);
    uint32_t tiles_per_batch = get_arg_val<uint32_t>(3);

    constexpr uint8_t cb_dat = 1;
    constexpr uint8_t cb_addr = 2;
    constexpr uint8_t cb_inVec = 3;
    constexpr uint8_t cb_collect = 4;

    constexpr uint32_t mat_page_size = get_tile_size(cb_dat);
    constexpr uint32_t vec_page_size = 1024;

    constexpr auto ell_val_args = TensorAccessorArgs<0, 5>();
    const auto ell_val_buf = TensorAccessor(ell_val_args, ell_val_base_addr, mat_page_size);

    constexpr auto ell_addr_args = TensorAccessorArgs<ell_val_args.next_compile_time_args_offset(), ell_val_args.next_common_runtime_args_offset()>();
    const auto ell_addr_buf = TensorAccessor(ell_addr_args, ell_addr_base_addr, mat_page_size);

    constexpr auto vec_args = TensorAccessorArgs<ell_addr_args.next_compile_time_args_offset(), ell_addr_args.next_common_runtime_args_offset()>();
    const auto vec_buf = TensorAccessor(vec_args, inVec_base_addr, vec_page_size);

    const uint32_t vecs_per_chunk = vec_page_size / 4 * vec_chunk_batch_size;

    DPRINT << "Ellpack matVec Rdcol ( " << first_tile_offset << ".+" << num_tiles << " in batches of " << tiles_per_batch << "), " << vec_chunks << " vec chunks" << ENDL();

    cb_reserve_back(cb_inVec, vec_chunk_batch_size); // Scratch: will never be pushed
    cb_reserve_back(cb_addr, tiles_per_batch); // scratch

    uint32_t end_tile = first_tile_offset + num_tiles;
    for (uint32_t batch_first_tile = first_tile_offset; batch_first_tile < end_tile; batch_first_tile += tiles_per_batch) {
        uint32_t end_tile_in_batch = std::min(num_tiles, batch_first_tile + tiles_per_batch);

        cb_reserve_back(cb_dat, tiles_per_batch); // only so we guarantee we can write_ptr + 4096 for each tile in there without buffer wrap-around
        cb_reserve_back(cb_collect, tiles_per_batch);


        auto val_wr_addr = get_write_ptr(cb_dat);
        auto addr_wr_addr = get_write_ptr(cb_addr);
        auto collect_wr_addr = get_write_ptr(cb_collect);

        {
            DeviceZoneScopedN("FetchingTiles");
            auto val_page_addr = val_wr_addr;
            auto addr_page_addr = addr_wr_addr;
            for (uint32_t i = batch_first_tile; i < end_tile_in_batch; ++i) {
                noc_async_read_page(i, ell_val_buf, val_page_addr);
                val_page_addr += mat_page_size;
                noc_async_read_page(i, ell_addr_buf, addr_page_addr);
                addr_page_addr += mat_page_size;
            }
        }

        uint32_t vec_chunk_offset = 0;
        for (uint32_t vec_chunk = 0; vec_chunk < vec_chunks; vec_chunk += vec_chunk_batch_size) {
            float* vec_ptr;
            {
                DeviceZoneScopedN("FetchingVecs");
                uint32_t vec_addr = get_write_ptr(cb_inVec);
                vec_ptr = reinterpret_cast<float*>(vec_addr);
                noc_async_read_page(vec_chunk, vec_buf, vec_addr);
                noc_async_read_barrier();
            }

            {
                DeviceZoneScopedN("Collecting");
                uint32_t* addr_page_addr = reinterpret_cast<uint32_t*>(addr_wr_addr);
                float* collect_page_addr = reinterpret_cast<float*>(collect_wr_addr);
                for (uint32_t t = 0; t < tiles_per_batch; ++t) {
                    collect_for_mul_tile(addr_page_addr, collect_page_addr, vec_ptr, vec_chunk_offset, vecs_per_chunk);
                    addr_page_addr += 1024;
                    collect_page_addr += 1024;
                }
            }

            vec_chunk_offset += vecs_per_chunk;
        }

        cb_push_back(cb_dat, tiles_per_batch);
        cb_push_back(cb_collect, tiles_per_batch);


        // float* ptr_a = reinterpret_cast<float*>(get_write_ptr(cb_a));
        // float* ptr_b = reinterpret_cast<float*>(get_write_ptr(cb_b));

    }

    DPRINT << "Rd Done" << ENDL();
}


