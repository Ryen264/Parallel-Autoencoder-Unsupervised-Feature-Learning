#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <string>
using std::string;

class Progress_Bar {
  int    _cur_step;
  int    _step;
  int    _size;
  int    _number_padding;
  string _text;
  string _progress;
  string _padding;

  static constexpr char _DONE_CHAR  = '=';
  static constexpr char _CUR_CHAR   = '>';
  static constexpr char _SPACE_CHAR = ' ';

public:
  /**
   * @brief Construct a new Progress_Bar object
   *
   * @param step The number of steps to display
   * @param text The text to display
   * @param size The length of the bar (default: 100 characters)
   */
  Progress_Bar(int step, const string &text = "", int size = 50);

  /**
   * @brief Update to the next step, first call starts at 1
   *
   */
  void update();
};

#endif