#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <chrono>
#include "file_swaps.hpp"

std::pair<int, int> getRandomIndices(int n)
{
   int i = std::rand() % n;
   int j = std::rand() % (n - 1);
   if (j >= i)
   {
      j++;
   }
   return std::make_pair(i, j);
}

void export_data_to_csv(std::vector<long double> data) {
   std::ofstream file;
   file.open("rowSwapTimesFile.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}

void printMatrix(std::vector<double> matrix, int rows, int cols)
{
   for (int i=0; i<rows; i++)
   {
      for (int j=0; j<cols; j++)
      {
         std::cout << matrix[i + j * rows] << " ";
      }
      std::cout << std::endl;
   }
   std::cout << std::endl;
}

int main(int argc, char *argv[])
{  
   const int num_trials = 3;
   const int max_dim = 16384;

   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsed_time = 0.L;

   std::string filename = "matrix.bin";

   std::vector<long double> trials;
   std::vector<long double> times;

   for (int N = 16; N <= max_dim; N *= 2)
   {  
      int size = N * N;
      // Generate the matrix
      std::vector<double> matrix(size);

      // init matrix elements in column major order
      for (int j=0; j<size; ++j)
      {
         matrix[j] = static_cast<double>(rand()) / RAND_MAX;
      }
      //printMatrix(matrix, N, N);

      // write the matrix to a file
      std::fstream file(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<char *>(&matrix[0]), size * sizeof(double));
      file.close();
      // Get random indices i and j for row swapping
      std::pair<int, int> rowIndices = getRandomIndices(N);
      int i = rowIndices.first;
      int j = rowIndices.second;
      //std::cout << i+1 << " --- " << j+1 << std::endl;

      for (int t=0; t<num_trials; t++)
      {
         // Open the file in read−write mode for swapping
         std::fstream fileToSwap(filename , std::ios::in | std::ios::out | std::ios::binary);
         
         // Measure the time required for row swapping using file I/O
         auto startTime = std::chrono::high_resolution_clock ::now();

         // Swap rows i and j in the file version of the matrix
         swapRowsInFile(fileToSwap, N, N, i, j);

         auto endTime = std::chrono::high_resolution_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

         elapsed_time = duration.count() * 1.e-9;
         trials.push_back(elapsed_time);
         /*
         std::vector<double> resultMatrix(size);
         fileToSwap.seekg(0);
         fileToSwap.read(reinterpret_cast<char *>(resultMatrix.data()), sizeof(double) * size);
         printMatrix(resultMatrix, N, N);
         */
         std::cout << "Matrix dimension: " << N << " x " << N << ", Elapsed time: " << elapsed_time << " Trial: " << t << std::endl;

         // Close the file after swapping
         fileToSwap.close();
      }

      long double averagePerformance = 0.0;

      for (int t=0; t<trials.size(); t++)
      {
         averagePerformance += trials[t];
      }

      times.push_back(averagePerformance);
      trials.clear();

      // after each problem size delete the test file
      std::remove(filename.c_str());
   }

   export_data_to_csv(times);

   return 0;
}