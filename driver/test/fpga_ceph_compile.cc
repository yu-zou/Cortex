#include "../ceph/semantic_fpga_backend.h"
#include <type_traits>
static_assert(std::is_base_of<cortex::SemanticBackend,
              cortex::SemanticFpgaBackend>::value, "actual executor backend interface");
