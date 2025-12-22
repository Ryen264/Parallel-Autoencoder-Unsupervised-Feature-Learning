#include "cpu_progress_bar.h"

Progress_Bar::Progress_Bar(int step, const string &text, int size)
    :_cur_step(0)
    ,_step(step)
    , _size(size)
    , _number_padding(1 + log10(step))
    , _text(text)
    , _progress(size, _DONE_CHAR)
    , _padding(size, _SPACE_CHAR) {};

void Progress_Bar::update() {
  if (_cur_step == _step) {
    printf(
        "\r%s %d/%d [%s] (100%%)", _text.c_str(), _cur_step, _step, _progress.c_str());
  } else {
    int percentage = round(100.0f * _cur_step / _step);
    int progress   = _cur_step * _size / _step;
    printf("\r%s %0*d/%d [%.*s%c%.*s] (%02d%%)",
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

  fflush(stdout);
}