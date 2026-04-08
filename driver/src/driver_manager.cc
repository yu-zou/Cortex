#include "smartssd_driver.h"
#include <mutex>

namespace cortex {

static SmartSSDDriver* g_driver_instance = nullptr;
static std::mutex       g_driver_mutex;

SmartSSDDriver& get_driver() {
    std::lock_guard<std::mutex> lk(g_driver_mutex);
    if (!g_driver_instance) {
        g_driver_instance = new SmartSSDDriver();
    }
    return *g_driver_instance;
}

void destroy_driver() {
    std::lock_guard<std::mutex> lk(g_driver_mutex);
    delete g_driver_instance;
    g_driver_instance = nullptr;
}

} // namespace cortex
