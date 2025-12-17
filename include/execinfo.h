/*
 * execinfo.h stub for Windows compatibility
 * 
 * This is a minimal stub implementation of execinfo.h for Windows.
 * The original execinfo.h is a Linux/POSIX header that provides backtrace functionality.
 * On Windows, these functions are stubbed as no-ops since backtrace is not available.
 */

#pragma once

#ifdef _WIN32

#ifdef __cplusplus
extern "C" {
#endif

// Stub implementation - returns 0 (no backtrace captured)
inline int backtrace(void** buffer, int size) { return 0; }

// Stub implementation - returns NULL
inline char** backtrace_symbols(void* const* buffer, int size) { return nullptr; }

// Stub implementation - no-op
inline void backtrace_symbols_fd(void* const* buffer, int size, int fd) {}

#ifdef __cplusplus
}
#endif

#endif