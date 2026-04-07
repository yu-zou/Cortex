#include "smartssd_driver.h"

namespace cortex {

static SmartSSDDriver* g_driver_instance = nullptr;

SmartSSDDriver& get_driver() {
    if (!g_driver_instance) {
        g_driver_instance = new SmartSSDDriver();
    }
    return *g_driver_instance;
}

void destroy_driver() {
    delete g_driver_instance;
    g_driver_instance = nullptr;
}

}
