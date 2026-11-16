#pragma once
#include "cortex_driver_api.h"
#include <atomic>

// Lives in the test executable, survives actual DSO dlclose. Not a DSO owner.
struct LoaderLeaseProbe {
  std::atomic<int> shutdowns{0}, unloads{0}, frees{0};
  std::atomic<int> sequence{0}, shutdown_order{0}, free_order{0}, unload_order{0};
  decltype(&cortex_semantic_read) read = nullptr;
};
