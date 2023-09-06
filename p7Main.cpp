#include "mem_swaps.hpp"
#include <chrono>
#include <fstream>
#include <random>
#include <iostream>
#include <utility> // For std::pair

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

std::vector<double> create_random_matrix(int size) {

   std::vector<double> matrix(size);

   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_real_distribution<double> dist(0,1);

   for (int i=0; i<size; i++) {
      matrix[i] = dist(rng);
   }
   return matrix;
}

void export_data_to_csv(std::vector<long double> data) {
   std::ofstream file;
   file.open("colSwapTimes.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}

std::vector<double> identity(int rows, int cols)
{
   int size = rows * cols;
   std::vector<double> matrix(size, 0.0);

   for (int i=0; i<rows; i++)
   {
      for (int j=0; j<cols; j++)
      {
         if (i==j) matrix[i + j * rows] = 1.0 * i + 1.0;
      }
   }

   return matrix;
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

int main()
{
   const int MAX_DIMENSION = 16384;
   const int num_trials = 3;

   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsed_time = 0.L;

   std::vector<long double> trials;
   std::vector<long double> data;

   for (int N = 16; N <= MAX_DIMENSION; N *= 2)
   {  
      std::pair<int, int> indices = getRandomIndices(N);
      int i = indices.first;
      int j = indices.second;

      for (int t = 0; t < num_trials; t++) 
      {
         std::vector<double> matrix = create_random_matrix(MAX_DIMENSION * MAX_DIMENSION);

         start = std::chrono::high_resolution_clock::now();
         swapCols(matrix, N, N, i, j);
         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

         elapsed_time = duration.count() * 1.e-9;
         trials.push_back(elapsed_time);

         std::cout << "Matrix dimension: " << N << " x " << N << ", Elapsed time: " << elapsed_time << " Trial: " << t << std::endl;
      }

      long double averagePerformance = 0.0;

      for (int t=0; t<trials.size(); t++)
      {
         averagePerformance += trials[t];
      }

      data.push_back(averagePerformance);
      trials.clear();
   }

   export_data_to_csv(data);

   return 0;
}