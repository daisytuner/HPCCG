#pragma once

#include <cstdint>
#include <compute_kernel_api/common.h>

#ifndef FACE_LAYOUT
#define FACE_LAYOUT 1
#endif

/**
 * Custom for 1-tile width
 */
static uint32_t faced_offset(uint32_t row, uint32_t col) {
    uint32_t in_face_row = row & 0xF;
    uint32_t face_id = ((row & 0x10) >> 3) | ((col & 0x10) >> 4);
    face_id += (row >> 5) * 4;
    uint32_t in_face_col = col & 0xF;

    return face_id * (16 * 16) + in_face_row * 16 + in_face_col;
}

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


    float* collect_row = collect_ptr;
    uint32_t* addr_row_ptr = addr_ptr;

    for (int rowIdx = 0; rowIdx < 32; ++rowIdx) {
        uint32_t* addr_row = addr_row_ptr;

        float* collect = collect_row;
        uint32_t* addr_entry = addr_row;
        bool line_done = false;
        for (int colIdx = 0; colIdx < 32; ++colIdx) {

            uint32_t adr = *addr_entry;

            // if (!line_done) {
                line_done = collect_for(collect, adr, vec_ptr, vec_chunk_offset, vec_chunk_end, rowIdx, colIdx);
            // }
            if (line_done) {
                break;
            }
            // *collect = 1.0f;

            if (colIdx == 15) {
                collect += NEXT_FACE_COL_OFFSET;
                addr_entry += NEXT_FACE_COL_OFFSET;
            } else {
                collect += 1;
                addr_entry += 1;
            }
        }
        if (rowIdx == 15) {
            collect_row += NEXT_FACE_ROW_OFFSET;
            addr_row_ptr += NEXT_FACE_ROW_OFFSET;
        } else {
            collect_row += NEXT_ROW_OFFSET;
            addr_row_ptr += NEXT_ROW_OFFSET;
        }
    }
}

static inline void unpacker_collect(
    uint8_t cb_addr,
    uint8_t cb_collect,
    uint8_t cb_vec,
    uint32_t vec_chunks,
    uint32_t vecs_per_chunk,
    uint32_t start_tile,
    uint32_t end_tile,
    uint32_t chunk_batch_size = 1,
    bool load_vecs = true,
    bool unload_vecs = true
) {
#ifdef TRISC_UNPACK
    uint32_t* addr_ptr = reinterpret_cast<uint32_t*>(CB_RD_PTR(cb_addr));
    float* collect_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_collect));
    
    DeviceZoneScopedN("Collect");

    for (uint32_t v = 0; v < vec_chunks; ++v) {
        if (load_vecs) {
            cb_wait_front(cb_vec, chunk_batch_size);
        }
        float* vec_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_vec));

        uint32_t* tile_addr_ptr = addr_ptr;
        float* tile_collect_ptr = collect_ptr;
        for (uint32_t i = start_tile; i < end_tile; ++i) {
            // DPRINT << "Collecting for tile " << i << ", v" << v << ENDL();
            collect_for_mul_tile(tile_addr_ptr, tile_collect_ptr, vec_ptr, v * vecs_per_chunk, vecs_per_chunk);

            tile_addr_ptr += 1024;
            tile_collect_ptr += 1024;
        }
        if (unload_vecs) {
            cb_pop_front(cb_vec, chunk_batch_size);
        }
    }
#endif
}