#include "progress_bar.h"
#include <cmath>
#include <cstdio>

Progress_Bar::Progress_Bar(int step, const string &text, int size)
    : _step(step)
    , _size(size)
    , _cur_step(1)
    , _text(text)
    , _number_padding(1 + log10(_step))
    , _progress(_DONE_CHAR, size)
    , _padding(_SPACE_CHAR, size) {};

void Progress_Bar::update() {
  if (_cur_step == _step) {
    printf(
        "%s %d/%d: [%s] (100%%)\n", _text.c_str(), _cur_step, _step, _progress.c_str());
  } else {
    int percentage = round(100.0f * _cur_step / _step);
    int progress   = _cur_step * _size / _step;
    printf("%s %0*d/%d: [%.*s%c%.*s] (%02d%%)\n",
           _text.c_str(),
           _number_padding,
           _cur_step,
           _step,
           progress,
           _progress.c_str(),
           _CUR_CHAR,
           _size - progress - 1,
           _padding.c_str(),
           percentage);

    ++_cur_step;
  }
}