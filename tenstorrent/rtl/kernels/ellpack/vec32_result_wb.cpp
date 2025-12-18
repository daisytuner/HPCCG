
#include <cstdint>

#include "dataflow_api.h"

#include "debug/dprint.h"

/**
 * Custom for 1-tile width
 */
constexpr uint32_t faced_offset(uint32_t row, uint32_t col) {
    uint32_t in_face_row = row & 0xF;
    uint32_t face_id = ((row & 0x10) >> 3) | ((col & 0x10) >> 4);
    face_id += (row >> 5) * 4;
    uint32_t in_face_col = col & 0xF;

    return face_id * (16 * 16) + in_face_row * 16 + in_face_col;
}

void kernel_main() {
    constexpr uint32_t unpack_diag = get_compile_time_arg_val(0);

    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page_vals = get_common_arg_val<uint32_t>(1);
    uint32_t page2tile_shift = get_common_arg_val<uint32_t>(2);

    uint32_t first_tile_offset = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 0;

    const uint32_t page_size = page_vals * sizeof(float);
    constexpr uint32_t write_size = 32 * sizeof(float);
    const uint32_t in_page_offset_mask = (1u << page2tile_shift) - 1;

    constexpr auto dest_args = TensorAccessorArgs<1, 3>();
    const auto dest = TensorAccessor(dest_args, dst_addr, page_size);

    DPRINT << "Ellpack Wb up ( " << first_tile_offset << ".+" << num_tiles << " pages (of " << page_vals << " vals), unpackDiag " << unpack_diag << ENDL();

    uint32_t end_tile = first_tile_offset + num_tiles;
    for (uint32_t tile = first_tile_offset; tile < end_tile; ++tile) {
        {
            DeviceZoneScopedN("WaitingForTiles");
            cb_wait_front(cb_out, 1);
        }

        DPRINT << "Wb got chunk " << tile << ENDL();

        auto rd_addr = get_read_ptr(cb_out);
        if constexpr (unpack_diag) {
            float* rd_ptr = reinterpret_cast<float*>(rd_addr);

            for (uint32_t d = 1; d < 32; ++d) { // 0,0 is already in the exact place we want it. First 32 floats contain 1 line of Face0 (0,0) and 2nd line of Face 0 (1,1), they are read before overwritten
                auto off = faced_offset(d, d);
                auto dat = rd_ptr[off];
                rd_ptr[d] = dat;
                // DPRINT << "[" << d << "] =" << dat << ENDL();
            }
        }

        {
            DeviceZoneScopedN("WbChunk");

            auto page = tile >> page2tile_shift; // idx of 32vals >> X = idx of page (however large that is)
            auto in_page_offset = (tile & in_page_offset_mask) * write_size;

            // DPRINT << " Wb tile " << tile << " to page " << page << " + offset " << in_page_offset << ENDL();

            noc_async_write_page(page, dest, rd_addr, write_size, in_page_offset);

            // float* ptr_a = reinterpret_cast<float*>(get_read_ptr(cb_out));

            // for (uint32_t x = 0; x < 32; ++x) {
            //     auto val_a = *(ptr_a+x);
            //     DPRINT << "[" << (x+tile*32) << "]=" << val_a << ENDL();
            // }

            noc_async_writes_flushed(); // all reads from SRAM are done now
        }

        cb_pop_front(cb_out, 1);

        // DPRINT << "Wb done chunk " << tile << ENDL();
    }

    noc_async_write_barrier();

    DPRINT << "Ellpack Wb done" << ENDL();
}
