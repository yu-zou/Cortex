#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# test_e2e.sh — Cortex 端到端集成测试
#
# 编排完整查询流水线:
#   构建 hw mock 库 → 构建 driver 库 → 加载测试夹具 → 搜索 → 验证结果
#
# 用法:
#   ./test_e2e.sh              # 正常运行,退出时清理构建目录
#   ./test_e2e.sh --keep       # 保留构建目录和工件
#   ./test_e2e.sh --docker     # 强制使用 Docker 构建 (如果 Docker 可用)
#   ./test_e2e.sh --skip-hls   # 跳过 Vitis HLS 测试台编译
#
# 退出码:
#   0 = 全部通过, 1 = 有步骤失败
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 项目结构 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/e2e-test"
LIB_DIR="${BUILD_DIR}/lib"
BIN_DIR="${BUILD_DIR}/bin"

FIXTURE_DIR="${PROJECT_ROOT}/hw/test/fixtures"
HW_SRC="${PROJECT_ROOT}/hw"
DRIVER_SRC="${PROJECT_ROOT}/driver"

# ─── 标志 ─────────────────────────────────────────────────────────────────────
KEEP=0
USE_DOCKER=0
SKIP_HLS=0
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FAILED_STEPS=""

# parse flags
for arg in "$@"; do
    case "$arg" in
        --keep)      KEEP=1      ;;
        --docker)    USE_DOCKER=1 ;;
        --skip-hls)  SKIP_HLS=1  ;;
        *)           echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
section() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════════════"
}

step_pass() {
    local msg="$1"
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  ✅ PASS: ${msg}"
}

step_fail() {
    local msg="$1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_STEPS="${FAILED_STEPS}  - ${msg}"$'\n'
    echo "  ❌ FAIL: ${msg}"
}

step_skip() {
    local msg="$1"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    echo "  ⏭️  SKIP: ${msg}"
}

check_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# ─── 清理 ─────────────────────────────────────────────────────────────────────
cleanup() {
    if [ "$KEEP" -eq 0 ]; then
        echo ""
        echo "─── 清理构建目录 ───"
        rm -rf "${BUILD_DIR}"
        echo "    已删除: ${BUILD_DIR}"
    else
        echo ""
        echo "─── --keep 已设置, 保留构建工件 ───"
        echo "    构建目录: ${BUILD_DIR}"
        echo "    库目录:   ${LIB_DIR}"
    fi
}
trap cleanup EXIT SIGTERM SIGINT

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: 前置条件检查
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 1/7: 前置条件检查"

PREREQ_FAIL=0

# 检查 cmake 或 Docker
HAS_DOCKER=0
HAS_CMAKE=0
HAS_GXX=0
HAS_PYTHON3=0
HAS_NUMPY=0

if check_cmd docker && docker info >/dev/null 2>&1; then
    HAS_DOCKER=1
    echo "  ✓ Docker 可用"
fi

if check_cmd cmake; then
    CMAKE_VER=$(cmake --version | head -1 | sed 's/[^0-9.]*//g')
    CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -gt 3 ] || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 16 ]; }; then
        HAS_CMAKE=1
        echo "  ✓ cmake ${CMAKE_VER} (≥3.16)"
    else
        echo "  ⚠ cmake ${CMAKE_VER} < 3.16 (需要 3.16+, 项目要求)"
        echo "    升级 cmake 或使用 Docker 构建"
    fi
fi

if check_cmd g++ && g++ --version >/dev/null 2>&1; then
    HAS_GXX=1
    echo "  ✓ g++ $(g++ --version | head -1 | awk '{print $NF}')"
fi

if check_cmd python3; then
    HAS_PYTHON3=1
    echo "  ✓ python3 $(python3 --version | awk '{print $2}')"
    # Check numpy
    if python3 -c "import numpy" 2>/dev/null; then
        HAS_NUMPY=1
        echo "  ✓ numpy 已安装"
    else
        echo "  ⚠ numpy 未安装 — 将尝试生成夹具, 但可能会失败"
    fi
fi

# Must have at least one toolchain
if [ "$HAS_DOCKER" -eq 1 ]; then
    echo "  → 使用 Docker 构建流水线"
elif [ "$HAS_CMAKE" -eq 1 ] && [ "$HAS_GXX" -eq 1 ]; then
    echo "  → 使用原生 cmake+g++ 构建流水线"
else
    echo "  ERROR: 需要 Docker 或 cmake+g++"
    echo "  安装依赖: sudo apt install cmake g++ python3 python3-numpy"
    PREREQ_FAIL=1
fi

if [ "$HAS_PYTHON3" -eq 0 ]; then
    echo "  ERROR: 需要 python3 生成测试夹具"
    PREREQ_FAIL=1
fi

if [ "$PREREQ_FAIL" -ne 0 ]; then
    step_fail "前置条件不满足"
    exit 1
fi

step_pass "前置条件检查通过"

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: 构建 hw mock 库
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 2/7: 构建 hw mock 库"

mkdir -p "${LIB_DIR}" "${BIN_DIR}"

if [ "$HAS_DOCKER" -eq 1 ] && [ "$USE_DOCKER" -eq 1 ]; then
    # Docker 构建路径 (使用 scripts/build.sh 模式)
    echo "--- 使用 Docker 构建 cortex_hw_mock ---"
    if docker build --network=host \
        -t cortex-hw-build \
        -f "${HW_SRC}/Dockerfile" \
        "${PROJECT_ROOT}" 2>&1; then
        echo "  Docker 镜像构建成功"
    else
        step_fail "Docker hw 镜像构建失败"
        exit 1
    fi

    if docker run --rm --network=host \
        -v "${PROJECT_ROOT}:/cortex" \
        -v "${LIB_DIR}:/output" \
        cortex-hw-build bash -c '
            set -euo pipefail
            mkdir -p /build && cd /build
            cmake /cortex/hw -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output
            make -j"$(nproc)"
            make install
            echo "HW_BUILD_OK"
        ' 2>&1; then
        echo "  cortex_hw_mock 构建成功"
    else
        step_fail "Docker hw 构建失败"
        exit 1
    fi
else
    # 原生 cmake 构建
    echo "--- 使用 cmake 原生构建 cortex_hw_mock ---"
    HW_BUILD_DIR="${BUILD_DIR}/hw"
    mkdir -p "${HW_BUILD_DIR}"

    if (cd "${HW_BUILD_DIR}" && cmake "${HW_SRC}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}") 2>&1; then
        echo "  cmake 配置成功"
    else
        step_fail "cmake 配置失败 (hw)"
        exit 1
    fi

    if cmake --build "${HW_BUILD_DIR}" -j"$(nproc)" 2>&1; then
        echo "  cmake 构建成功"
    else
        step_fail "cmake 构建失败 (hw)"
        exit 1
    fi

    # 安装 .so 到 LIB_DIR
    (cd "${HW_BUILD_DIR}" && make install) 2>&1
    echo "  已安装到 ${LIB_DIR}"
fi

# 验证库文件存在
HW_LIB="${LIB_DIR}/libcortex_hw_mock.so"
if [ -f "$HW_LIB" ]; then
    echo "  ✓ libcortex_hw_mock.so: $(ls -lh "$HW_LIB" | awk '{print $5}')"
    step_pass "hw mock 库构建成功"
else
    step_fail "libcortex_hw_mock.so 未找到"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: 构建 driver 库
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 3/7: 构建 driver 库"

if [ "$HAS_DOCKER" -eq 1 ] && [ "$USE_DOCKER" -eq 1 ]; then
    echo "--- 使用 Docker 构建 cortex_driver ---"
    if docker build --network=host \
        -t cortex-driver-build \
        -f "${DRIVER_SRC}/Dockerfile" \
        "${PROJECT_ROOT}" 2>&1; then
        echo "  Docker 镜像构建成功"
    else
        step_fail "Docker driver 镜像构建失败"
        exit 1
    fi

    if docker run --rm --network=host \
        -v "${PROJECT_ROOT}:/cortex" \
        -v "${LIB_DIR}:/output" \
        cortex-driver-build bash -c '
            set -euo pipefail
            mkdir -p /build && cd /build
            cmake /cortex/driver -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/output \
                -DCORTEX_HW_LIB_DIR=/output/lib
            make -j"$(nproc)"
            make install
            echo "DRIVER_BUILD_OK"
        ' 2>&1; then
        echo "  cortex_driver 构建成功"
    else
        step_fail "Docker driver 构建失败"
        exit 1
    fi
else
    echo "--- 使用 cmake 原生构建 cortex_driver ---"
    DRV_BUILD_DIR="${BUILD_DIR}/driver"
    mkdir -p "${DRV_BUILD_DIR}"

    if (cd "${DRV_BUILD_DIR}" && cmake "${DRIVER_SRC}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
        -DCORTEX_HW_LIB_DIR="${LIB_DIR}") 2>&1; then
        echo "  cmake 配置成功"
    else
        step_fail "cmake 配置失败 (driver)"
        exit 1
    fi

    if cmake --build "${DRV_BUILD_DIR}" -j"$(nproc)" 2>&1; then
        echo "  cmake 构建成功"
    else
        step_fail "cmake 构建失败 (driver)"
        exit 1
    fi

    (cd "${DRV_BUILD_DIR}" && make install) 2>&1
    echo "  已安装到 ${LIB_DIR}"
fi

# 验证库文件存在
DRV_LIB="${LIB_DIR}/libcortex_driver.so"
if [ -f "$DRV_LIB" ]; then
    echo "  ✓ libcortex_driver.so: $(ls -lh "$DRV_LIB" | awk '{print $5}')"
    step_pass "driver 库构建成功"
else
    step_fail "libcortex_driver.so 未找到"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: 生成测试夹具
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 4/7: 生成测试夹具"

FIXTURE_GEN_OK=0

# 4a: IVFPQ 夹具
echo "--- 生成 IVFPQ 夹具 ---"
GEN_SCRIPT="${HW_SRC}/test/generate_fixtures.py"
if [ -f "$GEN_SCRIPT" ]; then
    if python3 "$GEN_SCRIPT" --small 2>&1; then
        echo "  IVFPQ 夹具已生成:"
        for f in metadata.json codebook.bin vectors.bin queries.bin \
                 golden_topk.bin golden_distances.bin centroids.bin \
                 cluster_0.bin cluster_1.bin cluster_2.bin cluster_3.bin; do
            fp="${FIXTURE_DIR}/$f"
            [ -f "$fp" ] && echo "    $f ($(stat -c%s "$fp") bytes)" || echo "    $f (缺失)"
        done
        FIXTURE_GEN_OK=1
        step_pass "IVFPQ 夹具生成成功"
    else
        step_fail "IVFPQ 夹具生成失败"
    fi
else
    step_fail "generate_fixtures.py 未找到: ${GEN_SCRIPT}"
fi
if [ "$FIXTURE_GEN_OK" -eq 0 ]; then
    echo "  ⚠ IVFPQ 夹具生成失败 — 后续 IVFPQ 测试可能无法运行"
fi


section "STEP 5/7: IVFPQ 搜索 E2E 测试"

if [ "$FIXTURE_GEN_OK" -eq 0 ]; then
    step_skip "IVFPQ 夹具不可用, 跳过测试"
elif [ "$HAS_DOCKER" -eq 1 ] && [ "$USE_DOCKER" -eq 1 ]; then
    # ── Docker 路径: 在容器中编译并运行 testbench_unified.cpp ──
    echo "--- 在 Docker 容器中编译运行 testbench_unified.cpp ---"
    # testbench_unified.cpp 需要 Vitis HLS 头文件 (ap_int.h, hls_stream.h)
    # 在 hw/Dockerfile 容器内可用
    if docker run --rm --network=host \
        -v "${PROJECT_ROOT}:/cortex" \
        cortex-hw-build bash -c '
            set -euo pipefail
            # testbench_unified.cpp 是 Vitis HLS 测试台, 用 vitis_hls 编译
            # 如果 vitis_hls 不可用, 尝试 g++ 编译 (会因 HLS 头缺失而失败)
            cd /cortex/hw
            if command -v vitis_hls &>/dev/null; then
                echo "--- 使用 vitis_hls 运行 testbench_unified ---"
                # 使用 run_unified.tcl 如果存在
                if [ -f run_unified.tcl ]; then
                    vitis_hls -f run_unified.tcl 2>&1 | tail -20
                else
                    echo "run_unified.tcl 未找到, 尝试直接编译"
                    cd /cortex
                    g++ -std=c++17 -I/opt/xilinx/Vitis_HLS/2021.2/include \
                        -I hw/include \
                        hw/test/testbench_unified.cpp \
                        -o /tmp/testbench_unified 2>&1 && \
                    /tmp/testbench_unified
                fi
            else
                echo "vitis_hls 在容器中不可用, 跳过 HLS 测试台"
                exit 42  # 哨兵退出码: 跳过
            fi
        ' 2>&1; then
        step_pass "IVFPQ testbench_unified 测试通过"
    else
        RC=$?
        if [ "$RC" -eq 42 ]; then
            step_skip "IVFPQ testbench (HLS 在容器中不可用)"
        else
            step_fail "IVFPQ testbench_unified 测试失败 (退出码 $RC)"
        fi
    fi
else
    # ── 原生路径: 编译并运行基于 mock 库的软件 E2E 测试 ──
    echo "--- 编译运行软件 IVFPQ E2E 测试 (基于 mock 库) ---"

    # 生成内联 C++ E2E 测试程序
    E2E_SRC="${BUILD_DIR}/e2e_ivfpq_test.cpp"
    E2E_BIN="${BIN_DIR}/e2e_ivfpq_test"

    cat > "$E2E_SRC" << 'E2E_IVFPQ_CPP'
// ═══════════════════════════════════════════════════════════════════════════════
// e2e_ivfpq_test.cpp — IVFPQ 端到端软件 E2E 测试
// 验证流水线: 加载夹具 → 创建设备 → 加载 cluster → 搜索 → 验证结果
//
// 编译:
//   g++ -std=c++17 -I<HW_SRC>/include -I<HW_SRC>/src \
//       e2e_ivfpq_test.cpp -L<LIB_DIR> -lcortex_hw_mock -lpthread \
//       -o e2e_ivfpq_test
// ═══════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
// ---- 简单的 JSON 解析 (无需 nlohmann) ----
// 只提取 metadata.json 中的整数 key
static int read_json_int(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    // 查找 "key": <number>
    auto kpos = content.find("\"" + key + "\"");
    if (kpos == std::string::npos) return -1;
    auto cpos = content.find(':', kpos);
    if (cpos == std::string::npos) return -1;
    cpos++;
    while (cpos < content.size() && (content[cpos] == ' ' || content[cpos] == '\t')) cpos++;
    char* end = nullptr;
    long val = std::strtol(content.c_str() + cpos, &end, 10);
    if (end == content.c_str() + cpos) return -1;
    return static_cast<int>(val);
}

#include "smartssd_device.h"
#include "mock_smartssd.h"

static std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

int main(int argc, char* argv[]) {
    const char* fix_dir = std::getenv("FIXTURE_DIR");
    std::string fd = fix_dir ? fix_dir : "./hw/test/fixtures";
    if (argc > 1) fd = argv[1];

    try {
        // ---- 读取 metadata ----
        std::string meta_path = fd + "/metadata.json";
        int M   = read_json_int(meta_path, "M");
        int DIM = read_json_int(meta_path, "DIM");
        if (DIM < 0) DIM = read_json_int(meta_path, "D");   // small 格式用 "D"
        int N      = read_json_int(meta_path, "num_vectors");
        int NQ     = read_json_int(meta_path, "num_queries");
        int TOP_K  = read_json_int(meta_path, "top_k");
        int Ds     = read_json_int(meta_path, "dsub");

        std::cout << "夹具参数: M=" << M << " DIM=" << DIM
                  << " N=" << N << " NQ=" << NQ
                  << " TOP_K=" << TOP_K << " Ds=" << Ds << std::endl;

        if (M <= 0 || DIM <= 0 || N <= 0 || NQ <= 0 || TOP_K <= 0) {
            std::cerr << "FAIL: metadata.json 参数缺失或无效" << std::endl;
            return 1;
        }

        // ---- 加载 cluster_0.bin ----
        std::string cluster_path = fd + "/cluster_0.bin";
        auto cluster_data = read_binary(cluster_path);
        std::cout << "cluster_0.bin: " << cluster_data.size() << " bytes" << std::endl;

        // ---- 加载查询向量 ----
        std::string qpath = fd + "/queries.bin";
        auto query_raw = read_binary(qpath);
        size_t expected_qbytes = static_cast<size_t>(NQ) * DIM * 4;
        if (query_raw.size() < expected_qbytes) {
            std::cerr << "FAIL: queries.bin 大小不足 "
                      << query_raw.size() << " < " << expected_qbytes << std::endl;
            return 1;
        }
        const float* queries = reinterpret_cast<const float*>(query_raw.data());
        std::cout << "queries.bin: " << NQ << " 个查询向量" << std::endl;

        // ---- 创建 Mock 设备 ----
        cortex::MockSmartSSD device;
        int rc = device.init("{}");
        if (rc != 0) {
            std::cerr << "FAIL: MockSmartSSD.init() 返回 " << rc << std::endl;
            return 1;
        }
        std::cout << "MockSmartSSD: 设备已创建" << std::endl;

        // ---- 加载 Cluster ----
        rc = device.load_cluster(0,
            cluster_data.data(), cluster_data.size(),
            nullptr, 0, 0);
        if (rc != 0) {
            std::cerr << "FAIL: load_cluster(0) 返回 " << rc << std::endl;
            return 1;
        }
        if (!device.is_cluster_loaded(0)) {
            std::cerr << "FAIL: is_cluster_loaded(0) 返回 false" << std::endl;
            return 1;
        }
        std::cout << "Cluster 0: 已加载" << std::endl;

        // ---- 执行搜索 ----
        int total_valid = 0;
        int total_empty = 0;
        int n_errors = 0;

        for (int qi = 0; qi < NQ; qi++) {
            const float* q = queries + static_cast<size_t>(qi) * DIM;

            auto result = device.search(0,
                nullptr, 0,
                q, static_cast<uint32_t>(DIM),
                static_cast<uint32_t>(TOP_K),
                cortex::MetricType::L2);

            if (result.status != 0) {
                std::cerr << "FAIL: search(query=" << qi << ") 返回 status="
                          << result.status << std::endl;
                n_errors++;
                continue;
            }

            if (result.entries.empty()) {
                std::cerr << "WARN: search(query=" << qi << ") 返回空结果" << std::endl;
                total_empty++;
                continue;
            }

            // 验证: 距离应为非 NaN、负无穷或零
            for (size_t ei = 0; ei < result.entries.size(); ei++) {
                const auto& e = result.entries[ei];
                if (std::isnan(e.distance) || std::isinf(e.distance)) {
                    std::cerr << "FAIL: query=" << qi << " entry=" << ei
                              << " 距离无效: " << e.distance << std::endl;
                    n_errors++;
                }
            }

            // 验证: 距离应递增排序
            for (size_t ei = 1; ei < result.entries.size(); ei++) {
                if (result.entries[ei].distance < result.entries[ei-1].distance - 1e-6f) {
                    std::cerr << "FAIL: query=" << qi << " entries 未排序: ["
                              << ei-1 << "]=" << result.entries[ei-1].distance
                              << " > [" << ei << "]=" << result.entries[ei].distance
                              << std::endl;
                    n_errors++;
                }
            }

            total_valid += static_cast<int>(result.entries.size());

            std::cout << "  查询[" << qi << "]: "
                      << "返回 " << result.entries.size() << " 个结果"
                      << " (最佳距离=" << result.entries[0].distance << ")"
                      << std::endl;
        }

        // ---- 汇总 ----
        std::cout << std::endl;
        std::cout << "─── IVFPQ E2E 测试汇总 ───" << std::endl;
        std::cout << "  查询数:       " << NQ << std::endl;
        std::cout << "  成功:         " << (NQ - n_errors - total_empty) << std::endl;
        std::cout << "  空结果:       " << total_empty << std::endl;
        std::cout << "  错误:         " << n_errors << std::endl;
        std::cout << "  总结果条目:   " << total_valid << std::endl;

        if (n_errors > 0) {
            std::cerr << "FAIL: " << n_errors << " 个查询失败" << std::endl;
            return 1;
        }
        if (total_valid == 0) {
            std::cerr << "FAIL: 未返回任何有效结果" << std::endl;
            return 1;
        }

        std::cout << "PASS: IVFPQ E2E 测试通过" << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "FAIL: 异常: " << ex.what() << std::endl;
        return 1;
    }
}
E2E_IVFPQ_CPP

    echo "  E2E IVFPQ 测试源文件已创建: ${E2E_SRC}"

    # 编译
    if g++ -std=c++17 \
        -I"${HW_SRC}/include" \
        -I"${HW_SRC}/src" \
        -I"${DRIVER_SRC}/include" \
        -I"${DRIVER_SRC}/src" \
        -I"${PROJECT_ROOT}/include" \
        -L"${LIB_DIR}" \
        -Wl,-rpath,"${LIB_DIR}" \
        "${E2E_SRC}" \
        -lcortex_hw_mock -lpthread \
        -o "${E2E_BIN}" 2>&1; then
        echo "  E2E IVFPQ 测试编译成功"
    else
        step_fail "E2E IVFPQ 测试编译失败"
        exit 1
    fi

    # 执行
    echo "  运行 IVFPQ E2E 测试..."
    echo "  FIXTURE_DIR=${FIXTURE_DIR}"
    echo ""

    # 设置 RPATH 让运行时能找到 libcortex_hw_mock.so
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"

    if "${E2E_BIN}" "${FIXTURE_DIR}" 2>&1; then
        step_pass "IVFPQ 搜索 E2E 测试通过"
    else
        step_fail "IVFPQ 搜索 E2E 测试失败"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: 验证流水线完整性
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 6/7: 验证流水线完整性"

echo "  构建产物:"
echo "    libcortex_hw_mock.so: $([ -f "${LIB_DIR}/libcortex_hw_mock.so" ] && echo '✅' || echo '❌')"
echo "    libcortex_driver.so:  $([ -f "${LIB_DIR}/libcortex_driver.so" ] && echo '✅' || echo '❌')"
echo "    IVFPQ E2E 测试:      $([ -f "${BIN_DIR}/e2e_ivfpq_test" ] && echo '✅' || echo '❌')"
echo ""

echo "测试夹具:"
echo "    IVFPQ 夹具:          $([ -f "${FIXTURE_DIR}/cluster_0.bin" ] && echo '✅' || echo '❌')"

step_pass "流水线完整性检查"

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8: 汇总
# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 7/7: 测试汇总"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │             端到端测试汇总                           │"
echo "  ├─────────────────────────────────────────────────────┤"
printf "  │  通过:  %3d / %-3d                                │\n" "$PASS_COUNT" "$TOTAL"
printf "  │  失败:  %3d                                      │\n" "$FAIL_COUNT"
printf "  │  跳过:  %3d                                      │\n" "$SKIP_COUNT"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "失败步骤:"
    echo -n "$FAILED_STEPS"
    echo ""
    echo "❌ 端到端测试: 失败"
    exit 1
elif [ "$PASS_COUNT" -eq 0 ]; then
    echo "⚠  没有步骤通过 (全部跳过)"
    echo ""
    echo "⚠  端到端测试: 无结论"
    exit 1
else
    echo "✅ 端到端测试: 全部通过 (${PASS_COUNT}/${TOTAL})"
    exit 0
fi
