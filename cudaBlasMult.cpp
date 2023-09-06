#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void export_data_to_csv(std::vector<double> data) {
   std::ofstream file;
   file.open("cudaBlasPerf.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}

int main()
{  
   const int MAX_DIMENSION = 8192;
   const int num_trials = 5;

   std::vector<double> trials;
   std::vector<double> data;

   // Create cuBLAS handle
   cublasHandle_t handle;
   cublasCreate(&handle);

   for (int N = 16; N <= MAX_DIMENSION; N *= 2)
   {  
      double fp_op = 0.;
      fp_op = 2. * N * N * N + 2 * N * N;

      for (int i = 0; i < num_trials; i++)
      {
         // Allocate host memory for matrices
         double *h_A = new double[N * N];
         double *h_B = new double[N * N];
         double *h_C = new double[N * N];

         // Initialize input matrices
         for (int i = 0; i < N * N; ++i)
            h_A[i] = static_cast<double>(i % 100) / 100.0f;

         for (int i = 0; i < N * N; ++i)
            h_B[i] = static_cast<double>((i + 1) % 100) / 100.0f;

         for (int i = 0; i < N * N; ++i)
            h_C[i] = static_cast<double>((i + 1) % 100) / 100.0f;

         // Allocate device memory for matrices
         double *d_A, *d_B, *d_C;
         cudaMalloc((void **)&d_A, N * N * sizeof(double));
         cudaMalloc((void **)&d_B, N * N * sizeof(double));
         cudaMalloc((void **)&d_C, N * N * sizeof(double));

         // Copy input matrices from host to device
         cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
         cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);

         // Perform matrix multiplication
         double alpha = 1.0;
         double beta = 1.0;
         cudaEvent_t start, stop;
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
         cudaEventRecord(start, 0);
         cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
         cudaEventRecord(stop, 0);
         cudaEventSynchronize(stop);

         // Calculate elapsed time
         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);
         double elapsed_time = static_cast<double>(milliseconds / 1000);

         double performance = fp_op / 1.e6 / elapsed_time;
         trials.push_back(performance);

         // Print performance information
         std::cout << "Matrix multiplication performance:" << std::endl;
         std::cout << "Matrix size: " << N << "x" << N << "x" << N << std::endl;
         std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

         // Free device memory
         cudaFree(d_A);
         cudaFree(d_B);
         cudaFree(d_C);

         // Free host memory
         delete[] h_A;
         delete[] h_B;
         delete[] h_C;
      }
      double averagePerformance = 0.0;

      for (int i=0; i<trials.size(); i++)
      {
         averagePerformance += trials[i];
      }

      data.push_back(averagePerformance);
      trials.clear();
   }
   export_data_to_csv(data);
   // Destroy cuBLAS handle
   cublasDestroy(handle);

   return 0;
}