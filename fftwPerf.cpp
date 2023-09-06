#include <iostream>
#include <complex>
#include <cmath>
#include <fftw3.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

void export_data_to_csv(std::vector<long double> data) {
   std::ofstream file;
   file.open("fftwPerf.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();
}

int main()
{  
   const int MAX_DIMENSION = 512;
   const int num_trials = 3;

   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsed_time = 0.L;

   std::vector<long double> trials;
   std::vector<long double> data;

   for (int N = 16; N <= MAX_DIMENSION; N *= 2)
   {  
      double fp_op = 0.0;
      fp_op = 24.0 * N * N * N * log(N) + 6.0 * N * N * N;
      // Parameters
      int nx = N;     // Number of lattice points in x-direction
      int ny = N;     // Number of lattice points in y-direction
      int nz = N;     // Number of lattice points in z-direction
      double lx = 1.0; // Length of the cubic lattice in x-direction
      double ly = 1.0; // Length of the cubic lattice in y-direction
      double lz = 1.0; // Length of the cubic lattice in z-direction

      // Compute total number of lattice points
      int nxyz = nx * ny * nz;

      // Wave vector components in reciprocal lattice units
      double kx = 2.0 * M_PI / lx * 2; // Wave vector component in x-direction
      double ky = 2.0 * M_PI / ly * 3; // Wave vector component in y-direction
      double kz = 2.0 * M_PI / lz * 4; // Wave vector component in z-direction

      // Compute distances between spatial lattice sites
      double dx = lx / nx;
      double dy = ly / ny;
      double dz = lz / nz;

      for (int i = 0; i < num_trials; i++) 
      {  
         // Allocate memory for wave, fft_3, d_dx, d_dy, and d_dz arrays
         std::complex<double> *wave = new std::complex<double>[nxyz];
         std::complex<double> *fft_3 = new std::complex<double>[nxyz];
         std::complex<double> *d_dx = new std::complex<double>[nxyz];
         std::complex<double> *d_dy = new std::complex<double>[nxyz];
         std::complex<double> *d_dz = new std::complex<double>[nxyz];

         // Create forward and backward FFTW plans
         fftw_plan forward_plan = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(wave),
                                                   reinterpret_cast<fftw_complex *>(fft_3), FFTW_FORWARD, FFTW_ESTIMATE);
         fftw_plan backward_plan = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(fft_3),
                                                   reinterpret_cast<fftw_complex *>(fft_3), FFTW_BACKWARD, FFTW_ESTIMATE);

         // Generate complex plane wave on the cubic lattice
         for (int i = 0; i < nxyz; ++i)
         {
            int ix = i % nx;
            int iy = (i / nx) % ny;
            int iz = i / (nx * ny);
            double x = (ix - nx / 2) * dx;
            double y = (iy - ny / 2) * dy;
            double z = (iz - nz / 2) * dz;
            double phase = kx * x + ky * y + kz * z;

            wave[i] = std::polar(1.0, phase);
         }

         start = std::chrono::high_resolution_clock::now();
         // Perform forward FFT
         fftw_execute(forward_plan);

         // Copy the forward transform to wave 
         for (int i = 0; i < nxyz; ++i)
         {
            wave[i] = fft_3[i];
         }

         // Compute d/dx
         for (int i = 0; i < nxyz; ++i)
         {
            fft_3[i] = wave[i] * std::complex<double>(0.0, kx);
         }
         fftw_execute(backward_plan);
         for (int j = 0; j < nxyz; ++j)
         {
            d_dx[j] = fft_3[j] / static_cast<double>(nxyz);
         }

         // Compute d/dy
         for (int i = 0; i < nxyz; ++i)
         {
            fft_3[i] = wave[i] * std::complex<double>(0.0, ky);
         }
         fftw_execute(backward_plan);
         for (int j = 0; j < nxyz; ++j)
         {
            d_dy[j] = fft_3[j] / static_cast<double>(nxyz);
         }

         // Compute d/dz
         for (int i = 0; i < nxyz; ++i)
         {
            fft_3[i] = wave[i] * std::complex<double>(0.0, kz);
         }
         fftw_execute(backward_plan);
         for (int j = 0; j < nxyz; ++j)
         {
            d_dz[j] = fft_3[j] / static_cast<double>(nxyz);
         }

         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         
         elapsed_time = duration.count() * 1.e-9;

         long double performance = fp_op / 1.e6 / elapsed_time;
         trials.push_back(performance);

         // Clean up
         fftw_destroy_plan(forward_plan);
         fftw_destroy_plan(backward_plan);
         delete[] wave;
         delete[] fft_3;
         delete[] d_dx;
         delete[] d_dy;
         delete[] d_dz;
         
         std::cout << "Lattice size: " << N << ", Elapsed time: " << elapsed_time << " mflops: " << performance << std::endl;
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
