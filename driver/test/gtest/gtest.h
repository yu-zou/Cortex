#pragma once

namespace testing {
inline void InitGoogleTest(int*, char***) {}
inline int UnitTest() { return 0; }
}

#define RUN_ALL_TESTS() 0
