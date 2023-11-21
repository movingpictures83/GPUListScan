// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#include "GPUListScanPlugin.h"

void GPUListScanPlugin::input(std::string infile) {
  readParameterFile(infile);
}

void GPUListScanPlugin::run() {}

void GPUListScanPlugin::output(std::string outfile) {
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the list

  numElements = atoi(myParameters["N"].c_str());
  hostInput = (float*) malloc (numElements*sizeof(float));
  std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["data"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < numElements; ++i) {
        int k;
        myinput >> k;
        hostInput[i] = k;
 }
  cudaHostAlloc(&hostOutput, numElements * sizeof(float),
                cudaHostAllocDefault);

  // XXX the size is fixed for ease of implementation.
  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(float));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(float));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil((float)numElements / (BLOCK_SIZE << 1));
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxArray,
                              numElements);
  cudaDeviceSynchronize();
  scan<<<dim3(1, 1, 1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray,
                                    NULL, BLOCK_SIZE << 1);
  cudaDeviceSynchronize();
  fixup<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray,
                               numElements);

  cudaDeviceSynchronize();
        std::ofstream outsfile(outfile.c_str(), std::ios::out);
        int j;
        for (i = 0; i < numElements; ++i){
		outsfile << hostOutput[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);

  free(hostInput);
  cudaFreeHost(hostOutput);

}
PluginProxy<GPUListScanPlugin> GPUListScanPluginProxy = PluginProxy<GPUListScanPlugin>("GPUListScan", PluginManager::getInstance());

