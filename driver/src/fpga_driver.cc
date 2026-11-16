#include "cortex_fpga_driver_api.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
uint32_t u32(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint64_t u64(const uint8_t* p) { return u32(p) | uint64_t(u32(p + 4)) << 32; }
void put32(uint8_t* p, uint32_t v) { for (unsigned i=0;i<4;++i) p[i]=uint8_t(v>>(8*i)); }
void put64(uint8_t* p, uint64_t v) { put32(p,uint32_t(v));put32(p+4,uint32_t(v>>32)); }
float f32(const uint8_t* p) { uint32_t bits=u32(p);float v;std::memcpy(&v,&bits,4);return v; }
static_assert(sizeof(float)==4 && sizeof(cortex_topk_entry)==24, "supported ABI2 host layout");
struct State {
  std::string socket;
  unsigned timeout_ms=0, profile=0;
  std::atomic<bool> stopped{false};
  std::atomic_flag busy=ATOMIC_FLAG_INIT;
};
std::mutex lifecycle;
std::shared_ptr<State> current;
std::weak_ptr<State> retiring;
std::atomic<uint64_t> sequence{1};
struct Fd { int value=-1; ~Fd() { if(value>=0) ::close(value); } };
struct Call {
  std::shared_ptr<State> state;
  Clock::time_point deadline;
  cortex_fpga_cancel_fn canceled;
  void* context;
  int status() const {
    if(state->stopped.load(std::memory_order_acquire)) return -ESHUTDOWN;
    if(canceled && canceled(context)) return -ECANCELED;
    if(Clock::now()>=deadline) return -ETIMEDOUT;
    return 0;
  }
  int wait(int fd, short events) const {
    for(;;) {
      if(int r=status()) return r;
      auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
      pollfd p{fd,events,0};
      int n=::poll(&p,1,int(std::max<int64_t>(1,std::min<int64_t>(ms,10))));
      if(n<0) { if(errno==EINTR) continue;return -errno; }
      if(n && (p.revents&events)) return 0;
      if(n && (p.revents&(POLLERR|POLLHUP|POLLNVAL))) return -ECONNRESET;
    }
  }
  int transfer(int fd, uint8_t* data, size_t bytes, bool send) const {
    while(bytes) {
      if(int r=wait(fd,send?POLLOUT:POLLIN)) return r;
      ssize_t n=send ? ::send(fd,data,bytes,MSG_NOSIGNAL) : ::recv(fd,data,bytes,0);
      if(n<0) { if(errno==EINTR||errno==EAGAIN||errno==EWOULDBLOCK) continue;return -errno; }
      if(!n) return -ECONNRESET;
      data+=n;bytes-=size_t(n);
    }
    return status();
  }
};
int exchange_header(const Call& call,int fd,uint32_t op,uint64_t id,
                    uint64_t bytes,uint64_t max_reply,uint64_t& reply_bytes) {
  uint8_t h[CORTEX_FPGA_HEADER_BYTES]{};
  put32(h,CORTEX_FPGA_MAGIC);put32(h+4,CORTEX_FPGA_PROTOCOL_VERSION);
  put32(h+8,op);put32(h+12,call.state->profile);put64(h+16,id);put64(h+24,bytes);
  int r=call.transfer(fd,h,sizeof(h),true);
  if(r || op==CORTEX_FPGA_SEARCH) return r;
  r=call.transfer(fd,h,sizeof(h),false);
  if(r) return r;
  if(u32(h)!=CORTEX_FPGA_MAGIC || u32(h+4)!=CORTEX_FPGA_PROTOCOL_VERSION ||
     u32(h+8)!=(op|CORTEX_FPGA_REPLY) || u32(h+12)!=call.state->profile ||
     u64(h+16)!=id || u32(h+36) || u64(h+24)>max_reply) return -EPROTO;
  reply_bytes=u64(h+24);
  int32_t status;uint32_t bits=u32(h+32);std::memcpy(&status,&bits,4);
  if(status>0 || status < -4095 || (status && reply_bytes)) return -EPROTO;
  return status;
}
int connect_and_hello(const Call& call,Fd& fd,uint64_t id) {
  fd.value=::socket(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
  if(fd.value<0) return -errno;
  sockaddr_un address{};address.sun_family=AF_UNIX;
  std::memcpy(address.sun_path,call.state->socket.c_str(),call.state->socket.size()+1);
  if(::connect(fd.value,reinterpret_cast<sockaddr*>(&address),sizeof(address))) {
    if(errno!=EINPROGRESS) return -errno;
    if(int r=call.wait(fd.value,POLLOUT)) return r;
    int error=0;socklen_t len=sizeof(error);
    if(getsockopt(fd.value,SOL_SOCKET,SO_ERROR,&error,&len)) return -errno;
    if(error) return -error;
  }
  // The broker is an explicitly trusted same-user process, never an arbitrary
  // network endpoint. Its deployment owns device authorization and provenance.
  ucred peer{};socklen_t len=sizeof(peer);
  if(getsockopt(fd.value,SOL_SOCKET,SO_PEERCRED,&peer,&len)) return -errno;
  if(peer.uid!=geteuid()) return -EACCES;
  uint64_t bytes=0;
  if(int r=exchange_header(call,fd.value,CORTEX_FPGA_HELLO,id,0,8,bytes)) return r;
  if(bytes!=8) return -EPROTO;
  uint8_t reply[8];
  if(int r=call.transfer(fd.value,reply,8,false)) return r;
  if(u32(reply)!=CORTEX_FPGA_EXECUTION_HARDWARE || u32(reply+4)!=2) return -ENOTSUP;
  return 0;
}
int validate(const uint8_t* object,uint64_t bytes,const float* query,
             uint32_t dim,uint32_t k,uint32_t metric,uint32_t mode,uint32_t& count) {
  if(!object || !query || bytes<64 || bytes>CORTEX_FPGA_MAX_OBJECT_BYTES ||
     dim!=768 || !k || k>CORTEX_TOPK_MAX || metric>CORTEX_METRIC_IP || mode!=CORTEX_MODE_IVFPQ)
    return -EINVAL;
  count=u32(object+8);
  if(u32(object)!=96 || u32(object+4)!=768 || !count || count>8192 ||
     u32(object+12)!=0x31515043 || u32(object+16)!=2 || u32(object+20)!=256 ||
     u32(object+24)!=1 || bytes!=64+786432+3072+uint64_t(count)*112) return -EINVAL;
  for(unsigned i=28;i<64;++i) if(object[i]) return -EINVAL;
  for(unsigned i=0;i<dim;++i) if(!std::isfinite(query[i])) return -EINVAL;
  for(unsigned i=64;i<64+786432+3072;i+=4) if(!std::isfinite(f32(object+i))) return -EINVAL;
  std::vector<uint64_t> tokens;
  tokens.reserve(count);
  for(unsigned i=0;i<count;++i) {
    auto* row=object+64+786432+3072+size_t(i)*112;
    if(!u64(row+104) || u64(row+104)>UINT32_MAX) return -ERANGE;
    tokens.push_back(u64(row+96));
  }
  std::sort(tokens.begin(),tokens.end());
  if(std::adjacent_find(tokens.begin(),tokens.end())!=tokens.end()) return -EINVAL;
  return 0;
}
}

extern "C" uint32_t cortex_api_version() { return CORTEX_FPGA_DRIVER_ABI; }
extern "C" int cortex_driver_init(const char* config) {
  try {
    // Closed configuration grammar. Absolute socket path, explicit profile and
    // finite timeout are mandatory; reject extra fields and trailing bytes.
    if(!config) return -EINVAL;
    char path[108]{},profile[40]{};unsigned timeout=0;int end=0;
    if(std::sscanf(config,"{\"backend\":\"fpga\",\"socket\":\"%107[^\"]\",\"profile\":\"%39[^\"]\",\"timeout_ms\":%u}%n",
                   path,profile,&timeout,&end)!=3 || !end || config[end] || path[0]!='/' ||
       std::strpbrk(path,"\\\n\r\t") || !timeout || timeout>60000) return -EINVAL;
    char canonical[256];
    std::snprintf(canonical,sizeof(canonical),
      "{\"backend\":\"fpga\",\"socket\":\"%s\",\"profile\":\"%s\",\"timeout_ms\":%u}",path,profile,timeout);
    if(std::strcmp(config,canonical)) return -EINVAL;
    unsigned p=0;
    if(!std::strcmp(profile,"cpq1v2-residual-l2")) p=1;
    else if(!std::strcmp(profile,"cpq1v2-residual-ip")) p=2;
    else return -ENOTSUP;
    auto state=std::make_shared<State>();state->socket=path;state->profile=p;state->timeout_ms=timeout;
    std::lock_guard<std::mutex> lock(lifecycle);
    if(current || !retiring.expired()) return -EBUSY;
    Call call{state,Clock::now()+std::chrono::milliseconds(timeout),nullptr,nullptr};
    Fd fd;
    if(int r=connect_and_hello(call,fd,sequence.fetch_add(1))) return r;
    current=std::move(state);return 0;
  } catch(const std::bad_alloc&) { return -ENOMEM; } catch(...) { return -EIO; }
}
extern "C" void cortex_driver_shutdown() {
  std::lock_guard<std::mutex> lock(lifecycle);
  if(current) {
    current->stopped.store(true,std::memory_order_release);
    retiring=current;
  }
  current.reset();
}
extern "C" cortex_read_result cortex_fpga_read_cancellable(
    uint32_t cluster,const void* data,uint64_t bytes,const float* query,
    uint32_t dim,uint32_t k,uint32_t metric,uint32_t mode,
    cortex_fpga_cancel_fn canceled,void* context) {
  cortex_read_result out{};
  try {
    std::shared_ptr<State> state;
    { std::lock_guard<std::mutex> lock(lifecycle);state=current; }
    if(!state) { out.status=-ESHUTDOWN;return out; }
    Call call{state,Clock::now()+std::chrono::milliseconds(state->timeout_ms),canceled,context};
    auto run=[&]() -> int {
      if(int r=call.status()) return r;
      if(state->busy.test_and_set(std::memory_order_acquire)) return -EBUSY;
      struct Release { State& s;~Release(){s.busy.clear(std::memory_order_release);} } release{*state};
      uint32_t n=0;
      auto* object=static_cast<const uint8_t*>(data);
      if(int r=validate(object,bytes,query,dim,k,metric,mode,n)) return r;
      if(state->profile!=(metric==CORTEX_METRIC_L2 ? 1u : 2u)) return -ENOTSUP;
      if(int r=call.status()) return r;
      Fd fd;uint64_t id=sequence.fetch_add(1),ignored=0;
      if(int r=connect_and_hello(call,fd,id)) return r;
      std::vector<uint8_t> payload(32+size_t(dim)*4+size_t(bytes),0);
      put32(payload.data(),cluster);put32(payload.data()+4,dim);put32(payload.data()+8,k);
      put32(payload.data()+12,metric);put64(payload.data()+16,bytes);
      auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(call.deadline-Clock::now()).count();
      if(remaining<=0) return -ETIMEDOUT;
      put32(payload.data()+24,uint32_t(remaining));
      for(unsigned i=0;i<dim;++i) { uint32_t bits;std::memcpy(&bits,query+i,4);put32(payload.data()+32+4*i,bits); }
      std::memcpy(payload.data()+32+4*dim,object,size_t(bytes));
      if(int r=exchange_header(call,fd.value,CORTEX_FPGA_SEARCH,id,payload.size(),0,ignored)) return r;
      if(int r=call.transfer(fd.value,payload.data(),payload.size(),true)) return r;
      uint8_t header[40];
      if(int r=call.transfer(fd.value,header,40,false)) return r;
      if(u32(header)!=CORTEX_FPGA_MAGIC || u32(header+4)!=1 ||
         u32(header+8)!=(CORTEX_FPGA_SEARCH|CORTEX_FPGA_REPLY) ||
         u32(header+12)!=state->profile || u64(header+16)!=id || u32(header+36)) return -EPROTO;
      uint32_t status=u32(header+32);
      if(status) {
        if(status<uint32_t(-4095) || u64(header+24)) return -EPROTO;
        int32_t error;std::memcpy(&error,&status,4);return error;
      }
      const uint32_t count=std::min(k,n);
      if(u64(header+24)!=8+uint64_t(count)*16) return -EPROTO;
      std::vector<uint8_t> result(8+size_t(count)*16);
      if(int r=call.transfer(fd.value,result.data(),result.size(),false)) return r;
      if(u32(result.data())!=count || u32(result.data()+4)) return -EPROTO;
      for(unsigned i=0;i<count;++i) {
        const auto* row=result.data()+8+size_t(i)*16;
        float distance=f32(row);uint64_t token=u64(row+4);uint32_t length=u32(row+12);
        if(!std::isfinite(distance) || (metric==0 && distance<0)) return -EPROTO;
        if(i && (metric==0 ? distance<out.entries[i-1].distance : distance>out.entries[i-1].distance)) return -EPROTO;
        for(unsigned j=0;j<i;++j) if(out.entries[j].doc_addr==token) return -EPROTO;
        bool found=false;
        for(unsigned j=0;j<n;++j) {
          const auto* source=object+64+786432+3072+size_t(j)*112;
          if(u64(source+96)==token && u64(source+104)==length) {found=true;break;}
        }
        if(!found) return -EPROTO;
        out.entries[i].distance=distance;out.entries[i].doc_addr=token;out.entries[i].doc_length=length;
      }
      if(int r=call.status()) return r;
      out.count=count;return 0;
    };
    out.status=run();
  } catch(const std::bad_alloc&) { out.status=-ENOMEM; } catch(...) { out.status=-EIO; }
  if(out.status) { auto status=out.status;out={};out.status=status; }
  return out;
}
extern "C" cortex_read_result cortex_semantic_read(uint32_t cluster,const void* data,uint64_t bytes,
    const float* query,uint32_t dim,uint32_t k,uint32_t metric,uint32_t mode) {
  return cortex_fpga_read_cancellable(cluster,data,bytes,query,dim,k,metric,mode,nullptr,nullptr);
}
// No persistent cache: cluster identity never aliases another PG/object/version.
// This backend accepts immutable object bytes via SEARCH only, never Ceph writes.
extern "C" int cortex_semantic_write(uint32_t,const void*,uint64_t,const void*,uint64_t,uint32_t) { return -EOPNOTSUPP; }
extern "C" int cortex_is_cluster_cached(uint32_t) { return 0; }
extern "C" uint32_t cortex_get_last_cluster_id() { return 0; }
