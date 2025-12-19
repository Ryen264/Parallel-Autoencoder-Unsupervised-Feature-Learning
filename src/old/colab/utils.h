#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iomanip>
#include <sstream>
#include <cmath>
using namespace std;

/**
 * @brief Format time in milliseconds to string
 *
 * @param time_ms Time in milliseconds
 * @return string The formated string
 */
string format_time(float time_ms);

#endif