/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapids_logger/logger.hpp>

// Define log level (default to warn)
#ifndef CUML_LOG_ACTIVE_LEVEL
#define CUML_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_WARN
#endif

// Logging macros for cuML
#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define CUML_LOG_TRACE(...) ::ML::default_logger().trace(__VA_ARGS__)
#else
#define CUML_LOG_TRACE(...) void(0)
#endif

#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
#define CUML_LOG_DEBUG(...) ::ML::default_logger().debug(__VA_ARGS__)
#else
#define CUML_LOG_DEBUG(...) void(0)
#endif

#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO)
#define CUML_LOG_INFO(...) ::ML::default_logger().info(__VA_ARGS__)
#else
#define CUML_LOG_INFO(...) void(0)
#endif

#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN)
#define CUML_LOG_WARN(...) ::ML::default_logger().warn(__VA_ARGS__)
#else
#define CUML_LOG_WARN(...) void(0)
#endif

#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR)
#define CUML_LOG_ERROR(...) ::ML::default_logger().error(__VA_ARGS__)
#else
#define CUML_LOG_ERROR(...) void(0)
#endif

#if (CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL)
#define CUML_LOG_CRITICAL(...) ::ML::default_logger().critical(__VA_ARGS__)
#else
#define CUML_LOG_CRITICAL(...) void(0)
#endif
