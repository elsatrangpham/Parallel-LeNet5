#ifndef USE_CUDA

#include "filter.h"

#define TILE_WIDTH 8 // GCD(24, 8) in C1 and C3
#define NF (6 * 16 * 25) // in C3 (> C1)
__constant__ float c_d_filter[NF];

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess)                                \
    {                                                        \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
  }

__global__ void filter1(float *d_in, int channel_in, int height_in, int width_in,
                        float *d_out, int channel_out, int height_out, int width_out,
                        float *filter, int filterWidth, int W_grid,
                        float *d_bias)
{
  // Indices
  int n = blockIdx.x;                                       // sample index
  int m = blockIdx.y;                                       // channel out index
  int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y; // row index in output matrix
  int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x; // column index in output matrix

  int sample_strt = height_in * width_in * channel_in * n;      // sample index in input
  int h_in = h + filterWidth / 2;                               // row index in input
  int w_in = w + filterWidth / 2;                               // column index in input
  int filter_strt = channel_in * filterWidth * filterWidth * m; // filter index
  float acc = 0;                                                // pixel conv result
  // output index: (sample index) + (channel index) + (current channel)
  int i_out = (height_out * width_out * channel_out * n) + (height_out * width_out * m) + (h * width_out + w);

  // conv for the pixel in each channel_in
  for (int i_channel = 0; i_channel < channel_in; i_channel++)
  {
    for (int r = h_in - (filterWidth - 1) / 2; r <= h_in + (filterWidth - 1) / 2; r++)
    {
      for (int c = w_in - (filterWidth - 1) / 2; c <= w_in + (filterWidth - 1) / 2; c++)
      {
        // conv
        acc += d_in[r * width_in + c + sample_strt] * filter[filter_strt++];
      }
    }
    // Move to next channel in
    sample_strt += height_in * width_in;
  }
  // Bias adding
  acc += d_bias[m];

  // Final result
  d_out[i_out] = acc;

  // Check input images
  // int d_i = 0;
  // for (int col = 0; col < n; ++col)
  // {
  //   for (int row = 0; row < channel_in * height_in * width_in; ++row)
  //   {
  //     printf("%f ", d_in[d_i++]);
  //   }
  //   printf("\n");
  // }

  // Check input bias
  // for (int i = 0; i < channel_out; ++i)
  //   printf("%f\n", d_bias[i]);
}

int invoke_kernel(const float *h_in, int channel_in, int height_in, int width_in,
                  float *&h_out, int height_out, int width_out, int channel_out,
                  int n_sample, int filter_type, float *h_bias,
                  float *filter, int filterWidth, int stride, int pad_w, int pad_h)
{
  // TODO: Allocate device memories
  float *d_in, *d_out, *d_filter, *d_bias;
  size_t nBytes_d_in = height_in * width_in * channel_in * n_sample * sizeof(float);
  size_t nBytes_d_out = height_out * width_out * channel_out * n_sample * sizeof(float);
  size_t nBytes_d_filter = channel_in * filterWidth * filterWidth * channel_out * sizeof(float);
  size_t nBytes_d_bias = channel_out * sizeof(float);

  CHECK(cudaMalloc((void **)&d_in, nBytes_d_in));
  CHECK(cudaMalloc((void **)&d_out, nBytes_d_out));
  CHECK(cudaMalloc((void **)&d_filter, nBytes_d_filter));
  CHECK(cudaMalloc((void **)&d_bias, nBytes_d_bias));

  // TODO: Copy data to device memories
  CHECK(cudaMemcpy(d_in, h_in, nBytes_d_in, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_filter, filter, nBytes_d_filter, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bias, h_bias, nBytes_d_bias, cudaMemcpyHostToDevice));

  // TODO: Set grid size and call kernel
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  int W_grid = width_out / TILE_WIDTH;  // number of horizontal tiles per output map
  int H_grid = height_out / TILE_WIDTH; // number of vertical tiles per output map
  int Z_grid = H_grid * W_grid;
  dim3 gridSize(n_sample, channel_out, Z_grid);

  filter1<<<gridSize, blockSize>>>(d_in, channel_in, height_in, width_in,
                                     d_out, channel_out, height_out, width_out,
                                     d_filter, filterWidth, W_grid,
                                     d_bias);
  // Checks for synchronous errors
  cudaError_t errSync = cudaGetLastError();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));

  // TODO: Copy result from device memory
  CHECK(cudaMemcpy(h_out, d_out, nBytes_d_out, cudaMemcpyDeviceToHost));

  // TODO: Free device memories
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_filter));
  CHECK(cudaFree(d_bias));

  // cudaDeviceReset(); // Force to print

  // return filter type
  return 1;
}

#endif