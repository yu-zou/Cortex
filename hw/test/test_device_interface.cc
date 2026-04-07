#include <gtest/gtest.h>
#include <cstdint>

#include "smartssd_device.h"

namespace {

class StubDevice : public cortex::SmartSSDDevice {
public:
    int init(const char*) override { return 0; }
    void shutdown() override {}

    int load_cluster(uint32_t, const void*, uint64_t,
                     const void*, uint64_t, uint32_t) override {
        return 0;
    }

    cortex::SearchResult search(uint32_t, const void*, uint64_t,
                                const float*, uint32_t, uint32_t,
                                cortex::MetricType) override {
        return {};
    }

    bool is_cluster_loaded(uint32_t) const override { return false; }
    void evict_cluster(uint32_t) override {}
};

}

TEST(DeviceInterfaceTest, CanInstantiateViaBasePointer) {
    StubDevice stub;
    cortex::SmartSSDDevice* device = &stub;

    EXPECT_EQ(device->init("{}"), 0);
    device->shutdown();
}

TEST(DeviceInterfaceTest, MetricValuesMatchCAbi) {
    EXPECT_EQ(cortex::MetricType::L2, static_cast<cortex::MetricType>(0));
    EXPECT_EQ(cortex::MetricType::InnerProduct, static_cast<cortex::MetricType>(1));
}

TEST(DeviceInterfaceTest, SearchResultDefaultsAreEmptySuccess) {
    cortex::SearchResult result{};

    EXPECT_EQ(result.status, 0);
    EXPECT_TRUE(result.entries.empty());
}
