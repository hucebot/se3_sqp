#pragma once

#ifdef SQP_PROFILING
#include <chrono>
#include <string>

// RAII timer that accumulates time when profiling is enabled
class ScopedTimer {
public:
    ScopedTimer(double& accumulator)
        : _accumulator(accumulator),
          _start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        _accumulator += std::chrono::duration<double, std::milli>(end - _start).count();
    }

private:
    double& _accumulator;
    std::chrono::high_resolution_clock::time_point _start;
};

#define PROFILE_SCOPE(accumulator) ScopedTimer _timer_##__LINE__(accumulator)
#define PROFILE_DECLARE(name) double name = 0.0
#define PROFILE_PRINT(label, value) std::cout << "  " << label << ": " << value << " ms" << std::endl

#else
// When profiling disabled, everything compiles to nothing
#define PROFILE_SCOPE(accumulator) ((void)0)
#define PROFILE_DECLARE(name) ((void)0)
#define PROFILE_PRINT(label, value) ((void)0)

#endif
