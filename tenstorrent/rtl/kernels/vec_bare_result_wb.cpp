
#include <cstdint>

#include "dataflow_api.h"

#include "debug/dprint.h"

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

    constexpr auto dest_args = TensorAccessorArgs<0, 2>();
    const auto dest = TensorAccessor(dest_args, dst_addr, page_size);

    DPRINT << "Ellpack Wb up ( " << first_batch_offset << ".+" << num_batches << " pages, " << vecs_per_page << " vals/page)" << ENDL();

    uint32_t end_tile = first_batch_offset + num_batches;
    for (uint32_t tile = first_batch_offset; tile < end_tile; ++tile) {
        {
            DeviceZoneScopedN("WaitingForTiles");
            cb_wait_front(cb_out, 1);
        }

        DPRINT << "Wb got chunk " << tile << ENDL();

        {
            DeviceZoneScopedN("WbChunk");
            noc_async_write_page(tile, dest, get_read_ptr(cb_out));

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
