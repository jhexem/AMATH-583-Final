#include <vector>

void swapRows(std::vector<double> &matrix, int nRows, int nCols, int i, int j)
{
   for (int k=0; k<nCols; k++)
   {
      double temp = matrix[i + k * nCols];
      matrix[i + k * nCols] = matrix[j + k * nCols];
      matrix[j + k * nCols] = temp;
   }
}


void swapCols(std::vector<double> &matrix, int nRows, int nCols, int i, int j)
{  
   for (int k=0; k<nRows; k++)
   {
      double temp = matrix[k + i * nRows];
      matrix[k + i * nRows] = matrix[k + j * nRows];
      matrix[k + j * nRows] = temp;
   }
}
