#include "mock_smartssd.h"

extern "C" {

void* cortex_hw_create_device(void) {
    return new cortex::MockSmartSSD();
}

void cortex_hw_destroy_device(void* dev) {
    delete static_cast<cortex::MockSmartSSD*>(dev);
}

}
