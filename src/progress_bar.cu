#include "progress_bar.h"
#include <cmath>
#include <cstdio>

Progress_Bar::Progress_Bar(int step, const string &text, int size)
    : _step(step)
    , _size(size)
    , _cur_step(0)
    , _text(text)
    , _number_padding(1 + log10(step))
    , _progress(size, _DONE_CHAR)
    , _padding(size, _SPACE_CHAR) {};

void Progress_Bar::update() {
  if (_cur_step == _step) {
    printf(
        "\r%s %d/%d (100%%) [%s]", _text.c_str(), _cur_step, _step, _progress.c_str());
  } else {
    int percentage = round(100.0f * _cur_step / _step);
    int progress   = _cur_step * _size / _step;
    printf("\r%s %0*d/%d (%02d%%) [%.*s%c%.*s]",
           _text.c_str(),
           _number_padding,
           _cur_step,
           _step,
           percentage,
           progress,
           _progress.c_str(),
           _CUR_CHAR,
           _size - progress - 1,
           _padding.c_str());

    ++_cur_step;
  }

  fflush(stdout);
}