#ifndef SRC_LAYER_FILTER_H_
#define SRC_LAYER_FILTER_H_

#include <stdio.h>
#include <stdint.h>

int invoke_kernel(const float *h_in, int channel_in, int height_in, int width_in,
                   float *&h_out, int height_out, int width_out, int channel_out,
                   int n_sample, int filter_type, float* h_bias,
                   float *filter, int filterWidth, int stride, int pad_w, int pad_h);

#endif