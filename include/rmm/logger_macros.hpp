/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapids_logger/logger.hpp>

// Define log level (default to warn)
#ifndef RMM_LOG_ACTIVE_LEVEL
#define RMM_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_WARN
#endif

// Logging macros for RMM
#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define RMM_LOG_TRACE(...) ::RMM_NAMESPACE::default_logger().trace(__VA_ARGS__)
#else
#define RMM_LOG_TRACE(...) void(0)
#endif

#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
#define RMM_LOG_DEBUG(...) ::RMM_NAMESPACE::default_logger().debug(__VA_ARGS__)
#else
#define RMM_LOG_DEBUG(...) void(0)
#endif

#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO)
#define RMM_LOG_INFO(...) ::RMM_NAMESPACE::default_logger().info(__VA_ARGS__)
#else
#define RMM_LOG_INFO(...) void(0)
#endif

#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN)
#define RMM_LOG_WARN(...) ::RMM_NAMESPACE::default_logger().warn(__VA_ARGS__)
#else
#define RMM_LOG_WARN(...) void(0)
#endif

#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR)
#define RMM_LOG_ERROR(...) ::RMM_NAMESPACE::default_logger().error(__VA_ARGS__)
#else
#define RMM_LOG_ERROR(...) void(0)
#endif

#if (RMM_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL)
#define RMM_LOG_CRITICAL(...) ::RMM_NAMESPACE::default_logger().critical(__VA_ARGS__)
#else
#define RMM_LOG_CRITICAL(...) void(0)
#endif
