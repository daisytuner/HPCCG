// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/common.h>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "mat_vec_compute_parts.hpp"


#define HW_MODE_NONE 0
#define HW_MODE_FPU 1
#define HW_MODE_SFPU 2

#ifndef HW_MODE
#define HW_MODE HW_MODE_FPU
#endif

namespace NAMESPACE {



void compute_mat_mul(float* dat_ptr, uint32_t* addr_ptr, float* collect_ptr, float* res_ptr) {
    
    constexpr uint32_t NEXT_ROW_OFFSET = (FACE_LAYOUT == 1) ? 16 : 32;
    constexpr uint32_t NEXT_FACE_ROW_OFFSET = (FACE_LAYOUT == 1) ? (16*16 + 16) : 32;
    constexpr uint32_t NEXT_FACE_COL_OFFSET = (FACE_LAYOUT == 1) ? (16*16 - 15) : 1;

    float* collect_row = collect_ptr;
    float* dat_row = dat_ptr;
    for (int idx = 0; idx < 32; ++idx) { // row and col of result of matmul

        uint32_t* addr_row = addr_ptr + 32 * idx;

        float sum = 0.0f;
        float* dat_entry = dat_row;
        float* collect_entry = collect_row;
        for (int k = 0; k < 32; ++k) {
            bool face_left = k < 16;
            uint32_t adr = addr_row[k];
            if (adr != UINT32_MAX) {
                float mat_in = *dat_entry;
                float vec_in = *collect_entry;
                float elem_res = sum + mat_in * vec_in;

                DPRINT << "  [" << idx << "," << k << "]: " << sum << " + "  << mat_in << " * " << vec_in << "  => " << elem_res << ENDL();

                sum = elem_res;
                if (k == 15) {
                    collect_entry += NEXT_FACE_COL_OFFSET;
                    dat_entry += NEXT_FACE_COL_OFFSET;
                } else {
                    collect_entry += 1;
                    dat_entry += 1;
                }
            } else {
                break; // early abort, because right now it's left-aligned
            }
        }
        res_ptr[idx] = sum;

        if (idx == 15) {
            collect_row += NEXT_FACE_ROW_OFFSET;
            dat_row += NEXT_FACE_ROW_OFFSET;
        } else {
            collect_row += NEXT_ROW_OFFSET;
            dat_row += NEXT_ROW_OFFSET;
        }
    }
}

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

    // This semaphore seems to be left floating and only resets with the entire card. Its needed in cb_get_tile / cb_release_tile (and must start with 0)
    TTI_SEMINIT(1, 0, semaphore::t6_sem(semaphore::UNPACK_OPERAND_SYNC));

    binary_op_init_common(cb_dat, cb_dat, cb_collect);  // Unpack, Math, Pack
    add_tiles_init(cb_dat, cb_dat);

    UNPACK(DPRINT << "ellpack matVec up: " << num_tiles << " tiles (in batches of " << max_tile_batch_size << "), " << first_vec << " .. " << last_vec << " in chunks of " << vecs_per_chunk << ENDL());
    
    for (uint32_t first_tile_in_batch = 0; first_tile_in_batch < num_tiles; first_tile_in_batch += max_tile_batch_size) {
        DeviceZoneScopedN("Batch");
        uint32_t end_tile_in_batch = std::min(num_tiles, first_tile_in_batch + max_tile_batch_size);

        tile_regs_acquire();

        {
            DeviceZoneScopedN("WaitForCbTiles");
            cb_wait_front(cb_dat, max_tile_batch_size);
            cb_wait_front(cb_addr, max_tile_batch_size);
            cb_wait_front(cb_collect, max_tile_batch_size);
        }

        UNPACK({
            DPRINT << " CheckU 0x" << HEX() << CB_RD_PTR(cb_dat) << DEC() << ENDL();
        float* dat_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_dat));
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 16; ++i) {
                DPRINT << " CheckU  [" << j << "," << i << "] = " << *(dat_ptr+(j*16 + i)) << ENDL();
            }
        }
        });

        unpacker_collect(
            cb_addr,
            cb_collect,
            cb_vec,
            vecs_per_chunk,
            first_vec, last_vec,
            first_tile_in_batch, end_tile_in_batch
        );

        // UNPACK({
        //     if (first_tile_in_batch == 0) {
        //         float* collect_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_collect));
        //         float* vec_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_vec));
        //         for (uint32_t j = 0; j < 16; ++j) {
        //             for (uint32_t i = 0; i < 6; ++i) {
        //                 DPRINT << ", c[" << j << ", " << i << "] = " << collect_ptr[j*16+i] << ENDL();
        //             }
        //         }
        //     }
        // });

        UNPACK(DPRINT << "Unpack done" << ENDL());

        float* collect_ptr, *dat_ptr;
        uint32_t* addr_ptr;
        {
            // DeviceZoneScopedN("GetTileCollect");
            // UNPACK((llk_unpack_get_tile<false, true>(cb_collect, 0, (uint32_t*)&collect_ptr)));
            // PACK(llk_pack_get_tile(cb_collect, 0, (uint32_t*)&collect_ptr));
            cb_get_tile(cb_collect, 0, &collect_ptr);
        }

        {
            // DeviceZoneScopedN("GetTileDat");
            // UNPACK((llk_unpack_get_tile<false, true>(cb_dat, 0, (uint32_t*)&dat_ptr)));
            // PACK(llk_pack_get_tile(cb_dat, 0, (uint32_t*)&dat_ptr));
            cb_get_tile(cb_dat, 0, &dat_ptr);
        }
        {
            // DeviceZoneScopedN("GetTileAddr");
            // UNPACK((llk_unpack_get_tile<false, true>(cb_addr, 0, (uint32_t*)&addr_ptr)));
            // PACK(llk_pack_get_tile(cb_addr, 0, (uint32_t*)&addr_ptr));
            cb_get_tile(cb_addr, 0, &addr_ptr);
        }


        // Because for some magic and undocumented reason cb_get_tile does read_ptr-1, which is 1 L1 line before the actual data
        collect_ptr += 4;
        dat_ptr += 4;
        addr_ptr += 4;

        PACK(DPRINT << "Starting Pack batch" << ENDL());

        // now we have assembled mat-tiles in cb_collect matching each tile in cb_dat for mat_mul, where the first line

        {
            DeviceZoneScopedN("MatMul");
            for (uint32_t t = first_tile_in_batch; t < end_tile_in_batch; ++t) {
                cb_reserve_back(cb_res, 1);
                PACK(float* wr_ptr = reinterpret_cast<float*>(CB_WR_PTR(cb_res)));
                PACK(DPRINT << " MatMul tile " << t << ENDL());

                PACK(compute_mat_mul(dat_ptr, addr_ptr, collect_ptr, wr_ptr));

                // PACK(DPRINT << "Pushing result" << ENDL());
                cb_push_back(cb_res, 1);

                PACK(dat_ptr += 1024); // in float, not bytes
                PACK(addr_ptr += 1024);
                PACK(collect_ptr += 1024);
                PACK(wr_ptr += 32);
            }
        }

        tile_regs_wait();

        tile_regs_commit();

        cb_release_tile(cb_collect); // awkward. This will already block UNPACK until all 6 releases are done!
        cb_pop_front(cb_collect, max_tile_batch_size);

        cb_release_tile(cb_dat);
        cb_pop_front(cb_dat, max_tile_batch_size);

        cb_release_tile(cb_addr);
        cb_pop_front(cb_addr, max_tile_batch_size);

        // this will also necessarily wait for matmul to complete before

        
        tile_regs_release();
    }

    DPRINT << "Ellpack Compute done" << ENDL();
}

}  // namespace NAMESPACE
