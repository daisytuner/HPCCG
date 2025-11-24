// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/common.h>
#include <compute_kernel_api/eltwise_binary.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/matmul.h>
#include <compute_kernel_api/pack.h>
#include <tools/profiler/kernel_profiler.hpp>
#include "mat_vec_compute_parts.hpp"

#define HW_MODE_NONE 0
#define HW_MODE_FPU 1
#define HW_MODE_SFPU 2

#ifndef HW_MODE
#define HW_MODE HW_MODE_FPU
#endif

namespace NAMESPACE {

void MAIN {
    uint32_t vec_chunks = get_common_arg_val<uint32_t>(0);
    uint32_t vec_chunk_batch_size = get_common_arg_val<uint32_t>(1);
    bool stream_vec = get_common_arg_val<uint32_t>(2) != 0;

    uint32_t batches = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_batch = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);


    constexpr uint8_t cb_res = 0;
    constexpr uint8_t cb_dat = 1;
    constexpr uint8_t cb_addr = 2;
    constexpr uint8_t cb_vec = 3;
    constexpr uint8_t cb_collect = 4;

    constexpr uint32_t vec_page_size = 1024;
    constexpr uint32_t vecs_per_page = vec_page_size / 4;
    uint32_t vecs_per_chunk = vecs_per_page * vec_chunk_batch_size;
    constexpr uint32_t vecs_per_mat_tile = 32;

    // binary_op_init_common(cb_dat, cb_dat, cb_collect);  // Unpack, Math, Pack
    // add_tiles_init(cb_dat, cb_dat);

    mm_init(cb_dat, cb_collect, cb_res, 1);

    UNPACK(DPRINT << "ellpack matVec up: " << batches << " batch (" << tiles_per_batch << " tiles/batch), " << num_tiles << " tiles total, " << vec_chunks << "/" << vec_chunk_batch_size << " vec chunks" << ENDL());

    uint32_t tile = 0;
    for (uint32_t b = 0; b < batches; ++b) {
        DeviceZoneScopedN("Batch");
        uint32_t end_tile_in_batch = std::min(num_tiles, tile + tiles_per_batch);

        tile_regs_acquire();

        {
            UNPACK(DeviceZoneScopedN("WaitForCbTiles"));
            cb_wait_front(cb_dat, tiles_per_batch);
            cb_wait_front(cb_addr, tiles_per_batch);
            cb_wait_front(cb_collect, tiles_per_batch);
        }

        unpacker_collect(
            cb_addr,
            cb_collect,
            cb_vec,
            vec_chunks,
            vecs_per_chunk,
            tile, end_tile_in_batch,
            vec_chunk_batch_size,
            stream_vec || (b == 0),
            stream_vec || (b == (batches - 1))
        );

        UNPACK(DPRINT << "Unpack done" << ENDL());

        // UNPACK(float* dat_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_dat)));
        // UNPACK(float* col_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_collect)));
        // // UNPACK(dat_ptr += 16*16*2);
        // // UNPACK(col_ptr += 16*16*2);
        // for (int f = 0; f < 4; ++f) {
        //     for (int n = 0; n < 16; ++n) {
        //         UNPACK(DPRINT << " [" << n << "] ");
        //         for (int m = 0; m < 16; ++m) {
        //             UNPACK(auto dat_val = *dat_ptr++);
        //             UNPACK(DPRINT << dat_val << " ");
        //         }
        //         UNPACK(DPRINT << ENDL());
        //     }
        //     UNPACK(DPRINT << "---" << ENDL());
        // }
        // UNPACK(DPRINT << "###" << ENDL());
        // for (int f = 0; f < 4; ++f) {
        //     for (int n = 0; n < 16; ++n) {
        //         UNPACK(DPRINT << " [" << n << "] ");
        //         for (int m = 0; m < 16; ++m) {
        //             UNPACK(auto col_val = *col_ptr++);
        //             UNPACK(DPRINT << col_val << " ");
        //         }
        //         UNPACK(DPRINT << ENDL());
        //     }
        //     UNPACK(DPRINT << "---" << ENDL());
        // }

        auto tiles = end_tile_in_batch - tile;
        MATH(DPRINT << "Matmul " << tiles << " tiles" << ENDL());

        {
            MATH(DeviceZoneScopedN("Matmul"));
            UNPACK(DeviceZoneScopedN("Matmul"));
            for (uint32_t i = 0; i < tiles; ++i) {
                matmul_tiles(cb_dat, cb_collect, i, i, i, 1);
            }
        }

        tile_regs_commit();

        // MATH(dat_ptr += 16*16*2);
        // MATH(col_ptr += 16*16*2);

        tile_regs_wait();

        cb_pop_front(cb_collect, tiles_per_batch);
        cb_pop_front(cb_dat, tiles_per_batch);
        cb_pop_front(cb_addr, tiles_per_batch);

        PACK(DPRINT << "Pushing result" << ENDL());
        {
            PACK(DeviceZoneScopedN("Pack"));
            for (uint32_t i = 0; i < tiles; ++i) {
                cb_reserve_back(cb_res, 1);
                pack_tile(i, cb_res, 0);
                cb_push_back(cb_res, 1);
            }
        }

        
        tile_regs_release();
    }

    // DPRINT << "Ellpack Compute done" << ENDL();
}

}  // namespace NAMESPACE
