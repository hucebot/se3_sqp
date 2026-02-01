#pragma once

#include <common/types.h>

#include <cstdint>

/**
 * Variable types for trajectory optimization.
 * Standard types get O(1) lookup via array indexing.
 */
enum class VarType : uint8_t {
    CONF = 0,   // Configuration (nq)
    VEL,        // Velocity (nv)
    ACC,        // Acceleration (nv) - control input for inverse dynamics
    TORQ,       // Joint torques (nv)
    FORCE,      // Contact forces (use with contact name via allocate_force)
    _COUNT      // Sentinel for array sizing
};

/**
 * VarSlice represents a contiguous segment of the optimization vector.
 * Stores offset and size for fast Eigen segment access.
 */
struct VarSlice {
    int offset = 0;  // Start index in optimization vector
    int size = 0;    // Number of elements

    VarSlice() = default;
    VarSlice(int off, int sz) : offset(off), size(sz) {}

    // Segment accessors for Eigen vectors
    auto segment(VectorXdRef v) const { return v.segment(offset, size); }
    auto segment(VectorXdConstRef v) const { return v.segment(offset, size); }

    // Check if slice is valid (has been allocated)
    bool valid() const { return size > 0; }
};
