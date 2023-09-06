#include <vector>
#include <thread>
#include <iostream>

void sequentialTranspose(std::vector<int> &matrix, int rows, int cols)
{  
   std::vector<int> Tmatrix(rows * cols);

   for (int i=0; i<rows; i++)
   {
      for (int j=0; j<cols; j++)
      {  
         Tmatrix[j + i * cols] = matrix[i + j * rows];
      }
   }
   matrix = Tmatrix;
}

void testTranspose(std::vector<int> &matrix, int rows, int cols)
{  
   std::vector<int> Tmatrix(rows * cols);
   int size = rows * cols;

   for (int k=0; k<size; k++)
   {
      int i = k % rows;
      int j = (k - i) / rows;

      Tmatrix[j + i * cols] = matrix[i + j * rows];
      std::cout << "i is: " << i << " j is: " << j << " matrix[k] is: " << matrix[i + j * rows] << std::endl;
   }
   matrix = Tmatrix;
}

void partialTranspose(std::vector<int> &matrix, std::vector<int> &Tmatrix, int rows, int cols, int nthreads, int id)
{  
   int size = rows * cols;
   int start = (size / nthreads) * id;
   int end = (id == nthreads - 1) ? size : (size / nthreads) * (id + 1);

   for (int k=start; k<end; k++)
   {
      int i = k % rows;
      int j = (k - i) / rows;
      Tmatrix[j + i * cols] = matrix[i + j * rows];
   }
}

void threadedTranspose(std::vector<int> &matrix, int rows, int cols, int nthreads)
{  
   int size = rows * cols;

   std::vector<int> Tmatrix(size);
   std::vector<std::thread> threads(nthreads);
   
   for (int id=0; id<nthreads; id++)
   {
      threads[id] = std::thread(partialTranspose, std::ref(matrix), std::ref(Tmatrix), rows, cols, nthreads, id);
   }

   for (int id=0; id<nthreads; id++)
   {
      threads[id].join();
   }

   matrix = Tmatrix;
}