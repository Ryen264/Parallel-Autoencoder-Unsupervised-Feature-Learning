#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>

// Generate SVG bar chart for training times
void generate_bar_chart_svg(const std::vector<std::string>& labels, const std::vector<double>& times, const std::string& filename);

// Generate SVG line graph for speedup
void generate_speedup_graph_svg(const std::vector<std::string>& labels, const std::vector<double>& speedups, const std::string& filename);

// Generate HTML dashboard with embedded charts
void generate_visualization_html(const std::string& bar_chart_file, const std::string& speedup_file, const std::string& html_file);

#endif