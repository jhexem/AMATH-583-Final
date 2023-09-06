#include <iostream>
#include <chrono>
#include <cblas.h>
#include <fstream>
#include <vector>

void export_data_to_csv(std::vector<long double> data) {
   std::ofstream file;
   file.open("openBlasPerf.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}


int main()
{
   const int MAX_DIMENSION = 8192;
   const int num_trials = 5;
   const double alpha = 1.0;
   const double beta = 2.5;
   double A[MAX_DIMENSION * MAX_DIMENSION];
   double B[MAX_DIMENSION * MAX_DIMENSION];
   double C[MAX_DIMENSION * MAX_DIMENSION];

   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsed_time = 0.L;

   std::vector<long double> trials;
   std::vector<long double> data;

   for (int N = 16; N <= MAX_DIMENSION; N *= 2)
   {  
      double fp_op = 0.;
      fp_op = 2. * N * N * N + 2 * N * N;
      
      for (int i = 0; i < num_trials; i++) 
      {
         // Initialize matrices A and B with random values
         for (int j = 0; j < N * N; ++j)
         {
            A[j] = static_cast<double>(rand()) / RAND_MAX;
            B[j] = static_cast<double>(rand()) / RAND_MAX;
            C[j] = static_cast<double>(rand()) / RAND_MAX;
         }

         // Perform matrix multiplication and measure the execution time
         start = std::chrono::high_resolution_clock::now();
         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         
         elapsed_time = duration.count() * 1.e-9;
         if (elapsed_time == 0)
         {
            i--;
            continue;
         }
         long double performance = fp_op / 1.e6 / elapsed_time;
         trials.push_back(performance);

         std::cout << "Matrix dimension: " << N << " x " << N << ", Elapsed time: " << elapsed_time << " mflops/sec: " << performance << std::endl;
      }

      long double averagePerformance = 0.0;

      for (int i=0; i<trials.size(); i++)
      {
         averagePerformance += trials[i];
      }

      data.push_back(averagePerformance);
      trials.clear();
   }

   export_data_to_csv(data);

   return 0;
}