#include <fstream>
#include <vector>

void swapRowsInFile(std::fstream &file, int nRows, int nCols, int i, int j)
{  
   double temp1;
   double temp2;

   for (int k=0; k<nCols; k++)
   {
      file.seekg(sizeof(double) * (i + k * nCols));
      file.read(reinterpret_cast<char *>(&temp1), sizeof(double));

      file.seekg(sizeof(double) * (j + k * nCols));
      file.read(reinterpret_cast<char *>(&temp2), sizeof(double));

      file.seekp(sizeof(double) * (i + k * nCols));
      file.write(reinterpret_cast<char *>(&temp2), sizeof(double));

      file.seekp(sizeof(double) * (j + k * nCols));
      file.write(reinterpret_cast<char *>(&temp1), sizeof(double));
   }
}

void swapColsInFile(std::fstream &file, int nRows, int nCols, int i, int j)
{
   std::vector<double> temp1(nRows);
   std::vector<double> temp2(nRows);

   file.seekg(sizeof(double) * (i * nCols));
   file.read(reinterpret_cast<char *>(&temp1[0]), sizeof(double) * nRows);

   file.seekg(sizeof(double) * (j * nCols));
   file.read(reinterpret_cast<char *>(&temp2[0]), sizeof(double) * nRows);

   file.seekp(sizeof(double) * (i * nCols));
   file.write(reinterpret_cast<char *>(&temp2[0]), sizeof(double) * nRows);

   file.seekp(sizeof(double) * (j * nCols));
   file.write(reinterpret_cast<char *>(&temp1[0]), sizeof(double) * nRows);

}