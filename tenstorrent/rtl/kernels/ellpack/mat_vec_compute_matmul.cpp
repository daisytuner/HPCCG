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
    uint32_t max_tile_batch_size = get_common_arg_val<uint32_t>(0);
    uint32_t vecs_per_chunk = get_common_arg_val<uint32_t>(1);
    bool stream_vec = 1;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t first_vec = get_arg_val<uint32_t>(1); // actual offset to 1 vec
    uint32_t last_vec = get_arg_val<uint32_t>(2);


    constexpr uint8_t cb_res = 0;
    constexpr uint8_t cb_dat = 1;
    constexpr uint8_t cb_addr = 2;
    constexpr uint8_t cb_vec = 3;
    constexpr uint8_t cb_collect = 4;

    // binary_op_init_common(cb_dat, cb_dat, cb_collect);  // Unpack, Math, Pack
    // add_tiles_init(cb_dat, cb_dat);

    mm_init(cb_dat, cb_collect, cb_res, 1);

    UNPACK(DPRINT << "ellpack matVecMM up: " << num_tiles << " tiles (in batches of " << max_tile_batch_size << "), " << first_vec << " .. " << last_vec << " in chunks of " << vecs_per_chunk << ENDL());

    for (uint32_t first_tile_in_batch = 0; first_tile_in_batch < num_tiles; first_tile_in_batch += max_tile_batch_size) {
        DeviceZoneScopedN("Batch");
        uint32_t end_tile_in_batch = std::min(num_tiles, first_tile_in_batch + max_tile_batch_size);

        tile_regs_acquire();

        {
            UNPACK(DeviceZoneScopedN("WaitForCbTiles"));
            cb_wait_front(cb_dat, max_tile_batch_size);
            cb_wait_front(cb_addr, max_tile_batch_size);
            cb_wait_front(cb_collect, max_tile_batch_size);
        }

        // will fill cb_collect in memory, before the unpackers ever touch it for matmul
        unpacker_collect(
            cb_addr,
            cb_collect,
            cb_vec,
            vecs_per_chunk,
            first_vec, last_vec,
            first_tile_in_batch, end_tile_in_batch
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

        auto tiles = end_tile_in_batch - first_tile_in_batch;
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

        cb_pop_front(cb_collect, max_tile_batch_size);
        cb_pop_front(cb_dat, max_tile_batch_size);
        cb_pop_front(cb_addr, max_tile_batch_size);

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
