#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
using namespace std;

// Generate SVG bar chart for training times
void generate_bar_chart_svg(const vector<string>& labels, const vector<double>& times, const string& filename);

// Generate SVG line graph for speedup
void generate_speedup_graph_svg(const vector<string>& labels, const vector<double>& speedups, const string& filename);

// Generate HTML dashboard with embedded charts
void generate_visualization_html(const string& bar_chart_file, const string& speedup_file, const string& html_file);

#endif