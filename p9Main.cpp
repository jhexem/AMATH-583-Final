#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>

void export_data_to_csv(std::vector<double> data) {
   std::ofstream file;
   file.open("cudaDeviceToHostSpeed.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}

int main()
{  
   const int max_size = 268435456;

   std::vector<double> measurements;

   for (int dataSize = 8; dataSize <= max_size; dataSize *= 2)
   {
      int numDoubles = dataSize / sizeof(double);

      double *hostData = new double[numDoubles];

      for (int i=0; i<numDoubles; i++)
      {
         hostData[i] = static_cast<double>(i % 100) / 100.0;
      }

      double *deviceData;
      cudaMalloc((void **)&deviceData, dataSize);

      cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      cudaMemcpy(hostData, deviceData, dataSize, cudaMemcpyDeviceToHost);

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      double elapsed_time = static_cast<double>(milliseconds / 1000.0);

      double bandwidth = dataSize / elapsed_time;
      measurements.push_back(bandwidth);

      cudaFree(deviceData);
      delete[] hostData;

      std::cout << "Buffer size " << dataSize << " complete." << std::endl;
   }
   export_data_to_csv(measurements);

   return 0;
}