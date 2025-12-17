/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapids_logger/logger.hpp>

// Define log level (default to warn)
#ifndef RAFT_LOG_ACTIVE_LEVEL
#define RAFT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_WARN
#endif

// Logging macros for RAFT
#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define RAFT_LOG_TRACE(...) ::raft::default_logger().trace(__VA_ARGS__)
#else
#define RAFT_LOG_TRACE(...) void(0)
#endif

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
#define RAFT_LOG_DEBUG(...) ::raft::default_logger().debug(__VA_ARGS__)
#else
#define RAFT_LOG_DEBUG(...) void(0)
#endif

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO)
#define RAFT_LOG_INFO(...) ::raft::default_logger().info(__VA_ARGS__)
#else
#define RAFT_LOG_INFO(...) void(0)
#endif

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN)
#define RAFT_LOG_WARN(...) ::raft::default_logger().warn(__VA_ARGS__)
#else
#define RAFT_LOG_WARN(...) void(0)
#endif

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR)
#define RAFT_LOG_ERROR(...) ::raft::default_logger().error(__VA_ARGS__)
#else
#define RAFT_LOG_ERROR(...) void(0)
#endif

#if (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL)
#define RAFT_LOG_CRITICAL(...) ::raft::default_logger().critical(__VA_ARGS__)
#else
#define RAFT_LOG_CRITICAL(...) void(0)
#endif
