
#include <cstdint>

#include "dataflow_api.h"

#include "debug/dprint.h"

/**
 * Custom for 1-tile width
 */
uint32_t faced_offset(uint32_t row, uint32_t col) {
    uint32_t in_face_row = row & 0xF;
    uint32_t face_id = ((row & 0x10) >> 3) | ((col & 0x10) >> 4);
    face_id += (row >> 5) * 4;
    uint32_t in_face_col = col & 0xF;

    return face_id * (16 * 16) + in_face_row * 16 + in_face_col;
}

void kernel_main() {

    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t batch_size = get_common_arg_val<uint32_t>(1);

    uint32_t first_batch_offset = get_arg_val<uint32_t>(0);
    uint32_t num_batches = get_arg_val<uint32_t>(1);
    uint32_t first_tile_offset = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = 0;

    constexpr uint32_t page_size = 1024;
    constexpr uint32_t vecs_per_page = page_size / 4;
    constexpr uint32_t writes_per_page = vecs_per_page / 32;
    constexpr uint32_t write_size = 32 * sizeof(float); // only write one line of 32 floats

    constexpr auto dest_args = TensorAccessorArgs<0, 2>();
    const auto dest = TensorAccessor(dest_args, dst_addr, page_size);

    DPRINT << "Ellpack Diag Wb up ( " << first_tile_offset << ".+" << num_tiles << " lines)" << ENDL();

    uint32_t end_tile = first_tile_offset + num_tiles;
    uint32_t in_batch_offset = 0;
    for (uint32_t tile = first_tile_offset; tile < end_tile; ++tile) {
        {
            DeviceZoneScopedN("Pulling Tile");
            cb_wait_front(cb_out, 1);
        }

        uint32_t rd_addr = get_read_ptr(cb_out);
        float* rd_ptr = reinterpret_cast<float*>(rd_addr);

        for (uint32_t d = 1; d < 32; ++d) { // 0,0 is already in the exact place we want it. First 32 floats contain 1 line of Face0 (0,0) and 2nd line of Face 0 (1,1), they are read before overwritten
            auto off = faced_offset(d, d);
            auto dat = rd_ptr[off];
            rd_ptr[d] = dat;
            DPRINT << "[" << d << "] =" << dat << ENDL();
        }

        uint32_t b = tile >> 3; // convert from 32-entry lines to 256-entry pages
        uint32_t in_batch_tile = tile & 0x7;

        {
            DeviceZoneScopedN("Writing Tile");
            noc_async_write_page(b, dest, rd_addr, write_size, in_batch_tile * write_size); // one 32-entry line

            DPRINT << "Wb wrote line " << tile << " to " << in_batch_offset << " of batch " << b << ENDL();

        //  float* ptr_a = reinterpret_cast<float*>(get_read_ptr(cb_out));

        //  for (uint32_t x = 0; x < 32; ++x) {
        //      auto val_a = *(ptr_a+x);
        //      DPRINT << "[" << x << "]=" << val_a << ENDL();
        //  }

            noc_async_writes_flushed(); // all reads from SRAM are done now
        }

        cb_pop_front(cb_out, 1);

        // DPRINT << "Wb done chunk " << tile << ENDL();
    }

    noc_async_write_barrier();

    DPRINT << "Ellpack Wb done" << ENDL();
}
