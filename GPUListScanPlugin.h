#ifndef GPULISTSCANPLUGIN_H
#define GPULISTSCANPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUListScanPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};


#define BLOCK_SIZE 512 //@@ You can change this


__global__ void fixup(float *input, float *aux, int len) {
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
  if (blockIdx.x) {
    if (start + t < len)
      input[start + t] += aux[blockIdx.x - 1];
    if (start + BLOCK_SIZE + t < len)
      input[start + BLOCK_SIZE + t] += aux[blockIdx.x - 1];
  }
}

__global__ void scan(float *input, float *output, float *aux, int len) {
  // Load a segment of the input vector into shared memory
  __shared__ float scan_array[BLOCK_SIZE << 1];
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
  if (start + t < len)
    scan_array[t] = input[start + t];
  else
    scan_array[t] = 0;
  if (start + BLOCK_SIZE + t < len)
    scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
  else
    scan_array[BLOCK_SIZE + t] = 0;
  __syncthreads();

  // Reduction
  int stride;
  for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE)
      scan_array[index] += scan_array[index - stride];
    __syncthreads();
  }

  // Post reduction
  for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE)
      scan_array[index + stride] += scan_array[index];
    __syncthreads();
  }

  if (start + t < len)
    output[start + t] = scan_array[t];
  if (start + BLOCK_SIZE + t < len)
    output[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

  if (aux && t == 0)
    aux[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}

#endif
