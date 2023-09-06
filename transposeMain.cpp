#include "transpose.hpp"

void printMatrix(std::vector<int> matrix, int rows, int cols)
{
   for (int i=0; i<rows; i++)
   {
      for (int j=0; j<cols; j++)
      {
         std::cout << matrix[i + j * rows] << " ";
      }
      std::cout << std::endl;
   }
}

std::vector<int> createMatrix(int rows, int cols)
{  
   int size = rows * cols;
   std::vector<int> matrix(size);

   for (int i=0; i<size; i++)
   {
      matrix[i] = i;
   }
   
   return matrix;
}

void printRowMatrix(std::vector<int> &matrix, int rows, int cols)
{
   int size = rows * cols;

   for (int i=0; i<size; i++)
   {
      std::cout << matrix[i] << " ";
   }
   std::cout << std::endl;
}

int main()
{  
   int rows = 10;
   int cols = 20;
   int nthreads = 10;
   std::vector<int> matrix = createMatrix(rows, cols);

   printMatrix(matrix, rows, cols);
   std::cout << std::endl;
   printRowMatrix(matrix, rows, cols);

   sequentialTranspose(matrix, rows, cols);

   std::cout << std::endl;
   printMatrix(matrix, cols, rows);
   std::cout << std::endl;
   printRowMatrix(matrix, rows, cols);

   threadedTranspose(matrix, cols, rows, nthreads);

   std::cout << std::endl;
   printMatrix(matrix, rows, cols);
   std::cout << std::endl;
   printRowMatrix(matrix, rows, cols);

   return 0;
}