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

static void collect_for(
    float* result,
    uint32_t addr,
    float* vec_ptr,
    uint32_t vec_chunk_offset,
    uint32_t vec_chunk_offset_end,
    uint32_t rowIdx,
    uint32_t colIdx,
    bool* line_done,
    bool* behind_chunk
) {
    bool below_range = addr < vec_chunk_offset;
    bool above_range = addr >= vec_chunk_offset_end;

    if (addr == UINT32_MAX) {
        *line_done = true;
        return;
    } else if (above_range) {
        *behind_chunk = true;
        return;
    } else if (below_range) { // must not happen, because we should never make it to cols, for which we already passed the vec range
        DPRINT << " col [" << rowIdx << ", " << colIdx << "]: " << addr << " below vec_range" << vec_chunk_offset << ENDL();
        // DPRINT << " col [" << rowIdx << ", " << colIdx << "]: " << addr << " not in range" << ENDL();
    } else {
        auto val = vec_ptr[addr - vec_chunk_offset];
        *result = val;

        DPRINT << " col [" << rowIdx << ", " << colIdx << "] = " << val << " from " << addr << " | " << (addr - vec_chunk_offset) << ENDL();
    }

}

static inline void collect_for_mul_tile(
    uint32_t* addr_ptr,
    float* collect_ptr,
    uint8_t* next_col_ptr,
    bool first_run,
    float* vec_ptr,
    uint32_t vec_chunk_offset,
    uint32_t vec_chunk_end_offset
) {

    constexpr uint32_t NEXT_ROW_OFFSET = (FACE_LAYOUT == 1) ? 16 : 32;
    constexpr uint32_t NEXT_FACE_ROW_OFFSET = (FACE_LAYOUT == 1) ? (16*16 + 16) : 32;
    constexpr uint32_t NEXT_FACE_COL_OFFSET = (FACE_LAYOUT == 1) ? (16*16 - 15) : 1;
    constexpr uint8_t LAST_COL = 32;


    float* collect_row = collect_ptr;
    for (int rowIdx = 0; rowIdx < 32; ++rowIdx) {
        uint32_t* addr_row = addr_ptr + rowIdx * 32;

        float* collect;
        bool line_done = false;
        bool behind_chunk = false;
        uint8_t colIdx;
        if (first_run) {
            colIdx = 0;
            collect = collect_row;
        } else {
            colIdx = next_col_ptr[rowIdx];
            collect = collect_row + colIdx;
        }
        if (colIdx < LAST_COL) {
            // if (colIdx > 0) {
            //     DPRINT << " Res. row " << rowIdx << " at " << static_cast<uint32_t>(colIdx) << ENDL();
            // }
            for (; colIdx < LAST_COL; ++colIdx) {

                uint32_t adr = addr_row[colIdx];

                // if (!line_done) {
                    collect_for(collect, adr, vec_ptr, vec_chunk_offset, vec_chunk_end_offset, rowIdx, colIdx, &line_done, &behind_chunk);
                // }
                if (line_done) {
                    break;
                } else if (behind_chunk) {
                    break;
                }
                // *collect = 1.0f;

                if (colIdx == 15) {
                    collect += NEXT_FACE_COL_OFFSET;
                } else {
                    collect += 1;
                }
            }
            if (line_done) { // remember which cols we already checked and do not need to check again
                // DPRINT << " Row " << rowIdx << " done" << ENDL();
                next_col_ptr[rowIdx] = LAST_COL;
            } else {
                // DPRINT << " Row " << rowIdx << " next " << static_cast<uint32_t>(colIdx) << ENDL();
                next_col_ptr[rowIdx] = colIdx;
            }
        }

        if (rowIdx == 15) {
            collect_row += NEXT_FACE_ROW_OFFSET;
        } else {
            collect_row += NEXT_ROW_OFFSET;
        }
    }
}

static inline void unpacker_collect(
    uint8_t cb_addr,
    uint8_t cb_collect,
    uint8_t cb_vec,
    uint32_t vecs_per_chunk,
    uint32_t first_vec,
    uint32_t last_vec,
    uint32_t start_tile,
    uint32_t end_tile,
    bool load_vecs = true, // TODO, if we have all of the vector, we only need to walk over the tiles once
    bool unload_vecs = true
) {
#ifdef TRISC_UNPACK
    uint32_t* addr_ptr = reinterpret_cast<uint32_t*>(CB_RD_PTR(cb_addr));
    float* collect_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_collect));
    DeviceZoneScopedN("Collect");

    auto total_rows = (end_tile - start_tile) * 32;
    uint8_t row_progress[total_rows];
    bool first_run = true;

    // access vec_chunks aligned (that is what is in CB), but the first few vecs may be invalid if first_vec is not aligned
    // same at end. last vecs may be invalid if last_vec is not aligned.
    // but this should not matter, since those were the min / max vecs ever mentioned in the tiles processed.
    auto v_aligned_first = first_vec & ~(vecs_per_chunk - 1);
    for (uint32_t v = v_aligned_first; v < last_vec; v += vecs_per_chunk) {
        auto v_range_end = v + vecs_per_chunk;
        if (load_vecs) {
            cb_wait_front(cb_vec, 1);
        }
        float* vec_ptr = reinterpret_cast<float*>(CB_RD_PTR(cb_vec));

        uint32_t* tile_addr_ptr = addr_ptr;
        float* tile_collect_ptr = collect_ptr;
        uint8_t* row_progress_ptr = &row_progress[0];
        for (uint32_t i = start_tile; i < end_tile; ++i) {
            DPRINT << "Collecting for tile " << i << ", v" << v << ENDL();
            collect_for_mul_tile(tile_addr_ptr, tile_collect_ptr, row_progress_ptr, first_run, vec_ptr, v, v_range_end);

            tile_addr_ptr += 1024;
            tile_collect_ptr += 1024;
            row_progress_ptr += 32;
        }
        first_run = false;
        if (unload_vecs) {
            cb_pop_front(cb_vec, 1);
        }
    }
#endif
}