#include "visualization.h"

void generate_bar_chart_svg(const std::vector<std::string>& labels, const std::vector<double>& times, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    // SVG dimensions
    int width = 800;
    int height = 400;
    int margin_left = 150;
    int margin_right = 50;
    int margin_top = 50;
    int margin_bottom = 80;
    int chart_width = width - margin_left - margin_right;
    int chart_height = height - margin_top - margin_bottom;
    
    double max_time = *std::max_element(times.begin(), times.end());
    int bar_height = chart_height / labels.size() - 20;
    
    // SVG header
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    file << "  <style>\n";
    file << "    text { font-family: Arial, sans-serif; }\n";
    file << "    .title { font-size: 20px; font-weight: bold; }\n";
    file << "    .label { font-size: 14px; }\n";
    file << "    .value { font-size: 12px; }\n";
    file << "  </style>\n";
    
    // Title
    file << "  <text x=\"" << width/2 << "\" y=\"30\" class=\"title\" text-anchor=\"middle\">Training Time by Phase</text>\n";
    
    // Draw bars
    std::vector<std::string> colors = {"#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"};
    for (size_t i = 0; i < labels.size(); ++i) {
        int y = margin_top + i * (bar_height + 20);
        int bar_width = static_cast<int>((times[i] / max_time) * chart_width);
        
        // Draw bar
        file << "  <rect x=\"" << margin_left << "\" y=\"" << y 
             << "\" width=\"" << bar_width << "\" height=\"" << bar_height 
             << "\" fill=\"" << colors[i % colors.size()] << "\" opacity=\"0.8\"/>\n";
        
        // Label
        file << "  <text x=\"" << (margin_left - 10) << "\" y=\"" << (y + bar_height/2 + 5) 
             << "\" class=\"label\" text-anchor=\"end\">" << labels[i] << "</text>\n";
        
        // Value
        file << "  <text x=\"" << (margin_left + bar_width + 5) << "\" y=\"" << (y + bar_height/2 + 5) 
             << "\" class=\"value\">" << std::fixed << std::setprecision(2) << times[i] << "s</text>\n";
    }
    
    // X-axis
    file << "  <line x1=\"" << margin_left << "\" y1=\"" << (margin_top + chart_height) 
         << "\" x2=\"" << (margin_left + chart_width) << "\" y2=\"" << (margin_top + chart_height) 
         << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    file << "  <text x=\"" << (margin_left + chart_width/2) << "\" y=\"" << (height - 20) 
         << "\" class=\"label\" text-anchor=\"middle\">Time (seconds)</text>\n";
    
    file << "</svg>\n";
    file.close();
    std::cout << "Bar chart SVG saved to " << filename << std::endl;
}

void generate_speedup_graph_svg(const std::vector<std::string>& labels, const std::vector<double>& speedups, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    // SVG dimensions
    int width = 800;
    int height = 400;
    int margin_left = 100;
    int margin_right = 50;
    int margin_top = 50;
    int margin_bottom = 80;
    int chart_width = width - margin_left - margin_right;
    int chart_height = height - margin_top - margin_bottom;
    
    double max_speedup = *std::max_element(speedups.begin(), speedups.end());
    double y_scale = chart_height / (max_speedup * 1.2); // 20% padding
    double x_step = labels.size() > 1 ? chart_width / (labels.size() - 1) : 0;
    
    // SVG header
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    file << "  <style>\n";
    file << "    text { font-family: Arial, sans-serif; }\n";
    file << "    .title { font-size: 20px; font-weight: bold; }\n";
    file << "    .label { font-size: 14px; }\n";
    file << "    .value { font-size: 12px; }\n";
    file << "  </style>\n";
    
    // Title
    file << "  <text x=\"" << width/2 << "\" y=\"30\" class=\"title\" text-anchor=\"middle\">Cumulative Speedup</text>\n";
    
    // Draw axes
    file << "  <line x1=\"" << margin_left << "\" y1=\"" << margin_top 
         << "\" x2=\"" << margin_left << "\" y2=\"" << (margin_top + chart_height) 
         << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    file << "  <line x1=\"" << margin_left << "\" y1=\"" << (margin_top + chart_height) 
         << "\" x2=\"" << (margin_left + chart_width) << "\" y2=\"" << (margin_top + chart_height) 
         << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    
    // Y-axis label
    file << "  <text x=\"30\" y=\"" << (margin_top + chart_height/2) 
         << "\" class=\"label\" text-anchor=\"middle\" transform=\"rotate(-90 30 " 
         << (margin_top + chart_height/2) << ")\">Speedup (x)</text>\n";
    
    // X-axis label
    file << "  <text x=\"" << (margin_left + chart_width/2) << "\" y=\"" << (height - 20) 
         << "\" class=\"label\" text-anchor=\"middle\">Phase</text>\n";
    
    // Draw line and points
    std::stringstream path;
    path << "M ";
    for (size_t i = 0; i < labels.size(); ++i) {
        int x = margin_left + (labels.size() == 1 ? chart_width/2 : static_cast<int>(i * x_step));
        int y = margin_top + chart_height - static_cast<int>(speedups[i] * y_scale);
        
        if (i == 0) {
            path << x << "," << y;
        } else {
            path << " L " << x << "," << y;
        }
        
        // Draw point
        file << "  <circle cx=\"" << x << "\" cy=\"" << y 
             << "\" r=\"5\" fill=\"#ED7D31\" stroke=\"#C55A11\" stroke-width=\"2\"/>\n";
        
        // Draw label
        file << "  <text x=\"" << x << "\" y=\"" << (margin_top + chart_height + 20) 
             << "\" class=\"label\" text-anchor=\"middle\">" << labels[i] << "</text>\n";
        
        // Draw speedup value
        file << "  <text x=\"" << x << "\" y=\"" << (y - 10) 
             << "\" class=\"value\" text-anchor=\"middle\">" 
             << std::fixed << std::setprecision(2) << speedups[i] << "x</text>\n";
    }
    
    // Draw line
    if (labels.size() > 1) {
        file << "  <path d=\"" << path.str() << "\" stroke=\"#ED7D31\" stroke-width=\"3\" fill=\"none\"/>\n";
    }
    
    file << "</svg>\n";
    file.close();
    std::cout << "Speedup graph SVG saved to " << filename << std::endl;
}

void generate_visualization_html(const std::string& bar_chart_file, const std::string& speedup_file, const std::string& html_file) {
    std::ofstream file(html_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << html_file << std::endl;
        return;
    }
    
    file << "<!DOCTYPE html>\n";
    file << "<html>\n<head>\n";
    file << "  <meta charset=\"UTF-8\">\n";
    file << "  <title>Performance Analysis Dashboard</title>\n";
    file << "  <style>\n";
    file << "    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n";
    file << "    h1 { color: #333; text-align: center; }\n";
    file << "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n";
    file << "    .chart { margin: 30px 0; text-align: center; }\n";
    file << "    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: white; }\n";
    file << "    .footer { text-align: center; color: #666; margin-top: 30px; font-size: 14px; }\n";
    file << "    h2 { color: #555; }\n";
    file << "  </style>\n";
    file << "</head>\n<body>\n";
    file << "  <div class=\"container\">\n";
    file << "    <h1>🚀 Performance Analysis Dashboard</h1>\n";
    file << "    <div class=\"chart\">\n";
    file << "      <h2>Training Time by Phase</h2>\n";
    file << "      <img src=\"" << bar_chart_file << "\" alt=\"Bar Chart\">\n";
    file << "    </div>\n";
    if (!speedup_file.empty()) {
        file << "    <div class=\"chart\">\n";
        file << "      <h2>Cumulative Speedup</h2>\n";
        file << "      <img src=\"" << speedup_file << "\" alt=\"Speedup Graph\">\n";
        file << "    </div>\n";
    }
    file << "    <div class=\"footer\">\n";
    file << "      <p>Generated on: " << __DATE__ << " " << __TIME__ << "</p>\n";
    file << "      <p>Open the SVG files directly in your browser for interactive viewing</p>\n";
    file << "    </div>\n";
    file << "  </div>\n";
    file << "</body>\n</html>\n";
    file.close();
    std::cout << "\n✓ Visualization HTML dashboard saved to " << html_file << std::endl;
    std::cout << "  Open " << html_file << " in your browser to view the charts." << std::endl;
}
