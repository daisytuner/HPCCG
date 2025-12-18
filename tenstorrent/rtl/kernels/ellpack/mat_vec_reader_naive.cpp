
#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"

#include "debug/dprint.h"


void kernel_main() {
    uint32_t ell_val_base_addr = get_common_arg_val<uint32_t>(0);
    uint32_t ell_addr_base_addr = get_common_arg_val<uint32_t>(1);
    uint32_t inVec_base_addr = get_common_arg_val<uint32_t>(2);

    uint32_t tiles_per_batch = get_common_arg_val<uint32_t>(3); // how many tiles to push in one batch (because of CB limitations we need to use the same batch size all the time.)
    // and we do not want to use separate CB transactions per tile, as that would invalidate the old ptr and allow overwriting it
    uint32_t vecs_per_page = get_common_arg_val<uint32_t>(4);
    uint32_t vec2page_shift = get_common_arg_val<uint32_t>(5);
    uint32_t vec_chunk2page_shift = get_common_arg_val<uint32_t>(6);
    const uint32_t vecs_per_chunk = vecs_per_page << vec_chunk2page_shift;

    uint32_t first_tile_offset = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1); // how many tiles to read from the matrix
    uint32_t first_vec = get_arg_val<uint32_t>(2);
    uint32_t last_vec = get_arg_val<uint32_t>(3); // inclusive!

    constexpr uint8_t cb_dat = 1;
    constexpr uint8_t cb_addr = 2;
    constexpr uint8_t cb_inVec = 3;
    constexpr uint8_t cb_collect = 4;

    constexpr uint32_t mat_page_size = get_tile_size(cb_dat);
    const uint32_t vec_page_size = vecs_per_page * sizeof(float);

    constexpr auto ell_val_args = TensorAccessorArgs<0, 7>();
    const auto ell_val_buf = TensorAccessor(ell_val_args, ell_val_base_addr, mat_page_size);

    constexpr auto ell_addr_args = TensorAccessorArgs<ell_val_args.next_compile_time_args_offset(), ell_val_args.next_common_runtime_args_offset()>();
    const auto ell_addr_buf = TensorAccessor(ell_addr_args, ell_addr_base_addr, mat_page_size);

    constexpr auto vec_args = TensorAccessorArgs<ell_addr_args.next_compile_time_args_offset(), ell_addr_args.next_common_runtime_args_offset()>();
    const auto vec_buf = TensorAccessor(vec_args, inVec_base_addr, vec_page_size);

    DPRINT << "Ellpack matVec Rd ( " << first_tile_offset << ".+" << num_tiles << " in batches of " << tiles_per_batch << "), vecs " << first_vec << ".." << last_vec << " in chunks of " << vecs_per_chunk << ", page " << vec_page_size << ENDL();


    uint32_t end_tile = first_tile_offset + num_tiles;
    for (uint32_t tile = first_tile_offset; tile < end_tile; tile += tiles_per_batch) {
        uint32_t end_tile_in_batch = std::min(end_tile, tile + tiles_per_batch);

        {
            DeviceZoneScopedN("WaitingForCbSpace");
            cb_reserve_back(cb_dat, tiles_per_batch); // only so we guarantee we can write_ptr + 4096 for each tile in there without buffer wrap-around
            cb_reserve_back(cb_addr, tiles_per_batch);
            cb_reserve_back(cb_collect, tiles_per_batch);
        }


        {
            DeviceZoneScopedN("FetchingData");
            auto val_wr_addr = get_write_ptr(cb_dat);
            auto addr_wr_addr = get_write_ptr(cb_addr);
            for (uint32_t t = tile; t < end_tile_in_batch; ++t) {
                noc_async_read_page(t, ell_val_buf, val_wr_addr);
                // DPRINT << "Fetching dat -> 0x" << HEX() << val_wr_addr << DEC() << ENDL();
                val_wr_addr += mat_page_size;
                noc_async_read_page(t, ell_addr_buf, addr_wr_addr);
                addr_wr_addr += mat_page_size;
            }

            // iterate over aligned chunks of vectors. vec_first and vec_last may fall within the first / last chunk and will leave invalid data before / after. But keep alignment
            auto v_aligned_first = first_vec & ~(vecs_per_chunk - 1);
            for (uint32_t vec_aligned_start = v_aligned_first ; vec_aligned_start < last_vec; vec_aligned_start += vecs_per_chunk) {
                uint32_t vec_fetch_start = std::max(vec_aligned_start, first_vec);
                uint32_t vec_aligned_last = vec_aligned_start + vecs_per_chunk-1;
                uint32_t vec_fetch_last = std::min(vec_aligned_last, last_vec);

                DPRINT << "Fetch vecs " << vec_fetch_start << ".." << vec_fetch_last << " (win " << vec_aligned_start << ".." << vec_aligned_last << ")" << ENDL();
                uint32_t vec_chunk_page_idx = vec_aligned_start >> vec2page_shift;
                cb_reserve_back(cb_inVec, 1); // CB page size is 1 vec_chunk
                auto vec_chunk_ptr = get_write_ptr(cb_inVec);
                for (uint32_t vec_page_start = vec_fetch_start & ~(vecs_per_page - 1); vec_page_start < vec_fetch_last; vec_page_start += vecs_per_page) {
                    uint32_t page_idx = vec_page_start >> vec2page_shift;
                    auto page_in_chunk_idx = page_idx - vec_chunk_page_idx;
                    auto vec_page_ptr = vec_chunk_ptr + page_in_chunk_idx * vec_page_size;
                    // DPRINT << " Fetch vpage " << page_idx << " -> " << page_in_chunk_idx << ", *0x" << HEX() << vec_page_ptr << DEC() << ENDL();
                    noc_async_read_page(page_idx, vec_buf, vec_page_ptr);
                }
                noc_async_read_barrier();
                if (vec_aligned_start == v_aligned_first) { // we let the batch reads run concurrent with the first vecChunk, so mark them as available now
                    // DeviceZoneScopedN("PushingTiles");
                    // float* dat_ptr = reinterpret_cast<float*>(get_write_ptr(cb_inVec));
                    // for (int d = 0; d < 16; ++d) {
                    //     for (int e = 0; e < 16; ++e) {
                    //         DPRINT << " dat[" << d << "," << e << "] = " << dat_ptr[d*16+e] << ENDL();
                    //     }
                    // }
                    // float* vecf_ptr = reinterpret_cast<float*>(vec_chunk_ptr);
                    // DPRINT << "vec" << vec_fetch_start << ENDL();
                    // for (int i = 0; i < 512; ++i) {
                    //     DPRINT << "  ["<<i<<"]= " << vecf_ptr[i] << ENDL();
                    // }
    //                DPRINT << "a t" << tile << ":\n" << TileSlice(cb_dat, 0, SliceRange::h0_w0_32(), TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
    //                DPRINT << "b t" << tile << ":\n" << TileSlice(cb_addr, 0, SliceRange::h0_w0_32(), TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
                    cb_push_back(cb_dat, tiles_per_batch);
                    cb_push_back(cb_addr, tiles_per_batch);
                    cb_push_back(cb_collect, tiles_per_batch);
                }
                cb_push_back(cb_inVec, 1);
            }
        }



        // float* ptr_a = reinterpret_cast<float*>(get_write_ptr(cb_a));
        // float* ptr_b = reinterpret_cast<float*>(get_write_ptr(cb_b));

    }

    DPRINT << "Rd Done" << ENDL();
}


