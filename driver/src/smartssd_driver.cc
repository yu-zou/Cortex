#include "smartssd_driver.h"
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace cortex {

namespace {

struct ParsedConfig {
    uint32_t max_cache_slots = 64;
    std::string hw_lib_path;
};

ParsedConfig parse_config(const char* config_json) {
    ParsedConfig cfg;
    if (!config_json) return cfg;

    std::string json(config_json);

    auto extract_uint = [&](const std::string& key) -> int64_t {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return -1;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
        if (pos >= json.size()) return -1;
        return std::stoll(json.substr(pos));
    };

    auto extract_str = [&](const std::string& key) -> std::string {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return {};
        pos = json.find(':', pos);
        if (pos == std::string::npos) return {};
        pos = json.find('"', pos);
        if (pos == std::string::npos) return {};
        ++pos;
        auto end = json.find('"', pos);
        if (end == std::string::npos) return {};
        return json.substr(pos, end - pos);
    };

    auto slots = extract_uint("max_cache_slots");
    if (slots > 0) cfg.max_cache_slots = static_cast<uint32_t>(slots);

    cfg.hw_lib_path = extract_str("hw_lib_path");
    return cfg;
}

}

SmartSSDDriver::SmartSSDDriver() = default;

SmartSSDDriver::~SmartSSDDriver() {
    shutdown();
}

int SmartSSDDriver::init(const char* config_json) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (initialized_) {
        return 0;
    }

    ParsedConfig cfg;
    try {
        cfg = parse_config(config_json);
    } catch (...) {
        return -EINVAL;
    }

    max_cache_slots_ = cfg.max_cache_slots;

    std::string lib_path = cfg.hw_lib_path;
    if (lib_path.empty()) {
        lib_path = "libcortex_hw_mock.so";
    }

    hw_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!hw_handle_) {
        return -ENOENT;
    }

    using CreateFn = void* (*)();
    auto create_fn = reinterpret_cast<CreateFn>(dlsym(hw_handle_, "cortex_hw_create_device"));
    if (!create_fn) {
        dlclose(hw_handle_);
        hw_handle_ = nullptr;
        return -ENOSYS;
    }

    void* raw = create_fn();
    if (!raw) {
        dlclose(hw_handle_);
        hw_handle_ = nullptr;
        return -ENOMEM;
    }

    device_.reset(static_cast<SmartSSDDevice*>(raw));

    int rc = device_->init("{}");
    if (rc != 0) {
        device_.reset();
        dlclose(hw_handle_);
        hw_handle_ = nullptr;
        return rc;
    }

    initialized_ = true;
    return 0;
}

void SmartSSDDriver::shutdown() {
    std::lock_guard<std::mutex> lk(mutex_);

    if (!initialized_) return;

    lru_order_.clear();
    cache_map_.clear();
    reload_flags_.clear();

    if (device_) {
        device_->shutdown();
    }

    using DestroyFn = void (*)(void*);
    if (hw_handle_) {
        auto destroy_fn = reinterpret_cast<DestroyFn>(dlsym(hw_handle_, "cortex_hw_destroy_device"));
        if (destroy_fn && device_) {
            destroy_fn(device_.release());
        } else {
            device_.reset();
        }
        dlclose(hw_handle_);
        hw_handle_ = nullptr;
    } else {
        device_.reset();
    }

    initialized_ = false;
}

cortex_read_result SmartSSDDriver::semantic_read(
    uint32_t cluster_id, const void* object_data, uint64_t object_size,
    const float* query_vec, uint32_t query_dim, uint32_t top_k,
    uint32_t metric_type, uint32_t search_mode)
{
    std::lock_guard<std::mutex> lk(mutex_);

    cortex_read_result result{};
    result.status = 0;
    result.count = 0;

    if (!initialized_) {
        result.status = -EINVAL;
        return result;
    }

    auto it = cache_map_.find(cluster_id);
    if (it == cache_map_.end()) {
        int rc = device_->load_cluster(cluster_id, object_data, object_size, nullptr, 0, 0);
        if (rc != 0) {
            result.status = rc;
            return result;
        }

        if (cache_map_.size() >= max_cache_slots_) {
            evict_lru_locked();
        }

        lru_order_.push_front(cluster_id);
        cache_map_[cluster_id] = lru_order_.begin();
        reload_flags_[cluster_id] = false;
    } else {
        lru_order_.erase(it->second);
        lru_order_.push_front(cluster_id);
        cache_map_[cluster_id] = lru_order_.begin();
        reload_flags_[cluster_id] = false;
    }

    MetricType metric = (metric_type == CORTEX_METRIC_IP) ? MetricType::InnerProduct : MetricType::L2;
    SearchMode mode = (search_mode == CORTEX_MODE_HNSW) ? SearchMode::HNSW : SearchMode::IVFPQ;
    SearchResult sr = device_->search(cluster_id, object_data, object_size, query_vec, query_dim, top_k, metric, mode);

    result.status = sr.status;
    if (sr.status == 0) {
        uint32_t count = static_cast<uint32_t>(sr.entries.size());
        if (count > CORTEX_TOPK_MAX) count = CORTEX_TOPK_MAX;
        result.count = count;
        for (uint32_t i = 0; i < count; ++i) {
            result.entries[i].distance   = sr.entries[i].distance;
            result.entries[i].doc_addr   = sr.entries[i].doc_addr;
            result.entries[i].doc_length = sr.entries[i].doc_length;
            result.entries[i]._pad       = 0;
        }
    }

    last_cluster_id_ = cluster_id;
    return result;
}

int SmartSSDDriver::semantic_write(
    uint32_t cluster_id, const void* codebook, uint64_t codebook_size,
    const void* pq_payload, uint64_t payload_size, uint32_t vector_count)
{
    std::lock_guard<std::mutex> lk(mutex_);

    if (codebook_size == 0) return 0;  // no data to write

    if (!initialized_) return -EINVAL;

    int rc = device_->load_cluster(cluster_id, codebook, codebook_size, pq_payload, payload_size, vector_count);
    if (rc != 0) return rc;

    auto it = cache_map_.find(cluster_id);
    if (it == cache_map_.end()) {
        if (cache_map_.size() >= max_cache_slots_) {
            evict_lru_locked();
        }
        lru_order_.push_front(cluster_id);
        cache_map_[cluster_id] = lru_order_.begin();
        reload_flags_[cluster_id] = true;
    } else {
        lru_order_.erase(it->second);
        lru_order_.push_front(cluster_id);
        cache_map_[cluster_id] = lru_order_.begin();
        reload_flags_[cluster_id] = true;
    }

    return 0;
}

bool SmartSSDDriver::is_cluster_cached(uint32_t cluster_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return cache_map_.count(cluster_id) > 0;
}

bool SmartSSDDriver::was_codebook_reloaded(uint32_t cluster_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = reload_flags_.find(cluster_id);
    if (it == reload_flags_.end()) return false;
    return it->second;
}

uint32_t SmartSSDDriver::get_last_cluster_id() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_cluster_id_;
}

void SmartSSDDriver::evict_lru_locked() {
    if (lru_order_.empty()) return;
    uint32_t evict_id = lru_order_.back();
    lru_order_.pop_back();
    cache_map_.erase(evict_id);
    reload_flags_.erase(evict_id);
    device_->evict_cluster(evict_id);
}

} // namespace cortex
