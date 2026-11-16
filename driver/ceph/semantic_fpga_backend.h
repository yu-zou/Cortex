#pragma once
// Overlay for sys/src/cortex/osd, using the actual ABI2 executor interface.
#include "semantic_executor.h"
#include "cortex/loader/cortex_driver_loader.h"
#include "cortex/cortex_fpga_driver_api.h"
#include <dlfcn.h>
#include <cerrno>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cortex {
class SemanticFpgaBackend final : public SemanticBackend {
  CortexDriverLoader::Lease owner;
  decltype(&cortex_fpga_read_cancellable) read = nullptr;
 public:
  explicit SemanticFpgaBackend(CortexDriverLoader::Lease lease)
      : owner(std::move(lease)) {
    if(!owner || owner.startup_config().compare(0,18,"{\"backend\":\"fpga\","))
      throw std::invalid_argument("explicit FPGA configuration required");
    Dl_info info{};
    if(!dladdr(reinterpret_cast<void*>(owner.api().semantic_read),&info))
      throw std::invalid_argument("unmapped FPGA driver");
    void* handle=dlopen(info.dli_fname,RTLD_NOW|RTLD_NOLOAD);
    if(!handle) throw std::invalid_argument("missing FPGA driver owner");
    read=reinterpret_cast<decltype(read)>(dlsym(handle,"cortex_fpga_read_cancellable"));
    Dl_info extension{};
    bool same=read && dladdr(reinterpret_cast<void*>(read),&extension) &&
      extension.dli_fbase==info.dli_fbase;
    dlclose(handle);
    if(!same) throw std::invalid_argument("missing owned FPGA cancellation API");
  }
  // Authorized deployment selects the separately reviewed bounded socket DSO.
  // Never inherit CPU hash attestation or infer this from ABI2 alone.
  bool finite_return_contract() const noexcept override { return true; }
  int run(const SemanticInput& in,SemanticResult& out,
          const std::atomic<bool>& canceled) override {
    out.count=0;
    if(canceled.load(std::memory_order_acquire)) return -ECANCELED;
    size_t k=in.requested_top_k ? in.requested_top_k : in.result_capacity;
    if(!k || k>CORTEX_TOPK_MAX || k>out.entries.size() || k>in.result_capacity ||
       in.query.size()!=768) return -EINVAL;
    auto cancel=[](void* context) -> int {
      return static_cast<const std::atomic<bool>*>(context)->load(std::memory_order_acquire);
    };
    auto result=read(in.cache_id,in.object.data(),in.object.size(),in.query.data(),
        uint32_t(in.query.size()),uint32_t(k),in.metric,in.search_mode,
        cancel,const_cast<std::atomic<bool>*>(&canceled));
    if(result.status) return result.status<0 ? result.status : -EIO;
    if(result.count>k) return -EOVERFLOW;
    for(uint32_t i=0;i<result.count;++i) {
      const auto& e=result.entries[i];
      if(e._pad || !std::isfinite(e.distance)) return -EIO;
      out.entries[i]={e.doc_addr,e.doc_length,e.distance};
    }
    if(canceled.load(std::memory_order_acquire)) return -ECANCELED;
    out.count=result.count;return 0;
  }
};
} // namespace cortex
