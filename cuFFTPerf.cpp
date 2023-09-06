#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

__device__ cuDoubleComplex complexToCuComplex(double real, double imag)
{
    return make_cuDoubleComplex(real, imag);
}

__device__ void cuComplexToComplex(cuDoubleComplex c, double* real, double* imag)
{
    *real = cuCreal(c);
    *imag = cuCimag(c);
}

__device__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
                                cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b));
}

__device__ cuDoubleComplex operator/(cuDoubleComplex a, double b)
{
    return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__global__ void computeDdx(cuDoubleComplex* wave_gpu, cuDoubleComplex* fft_3_gpu, double kx, int nxyz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nxyz) {
        cuDoubleComplex factor = make_cuDoubleComplex(0.0, kx);
        fft_3_gpu[idx] = wave_gpu[idx] * factor;
    }
}

__global__ void computeScale(cuDoubleComplex* fft_3_gpu, cuDoubleComplex* d_dx_gpu, double scale, int nxyz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nxyz) {
        d_dx_gpu[idx] = fft_3_gpu[idx] / scale;
    }
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
         cuDoubleComplex* wave = new cuDoubleComplex[nxyz];
         cuDoubleComplex* fft_3 = new cuDoubleComplex[nxyz];
         cuDoubleComplex* d_dx = new cuDoubleComplex[nxyz];
         cuDoubleComplex* d_dy = new cuDoubleComplex[nxyz];
         cuDoubleComplex* d_dz = new cuDoubleComplex[nxyz];

         // Allocate memory on the GPU for wave, fft_3, d_dx, d_dy, and d_dz arrays
         cuDoubleComplex* wave_gpu;
         cuDoubleComplex* fft_3_gpu;
         cuDoubleComplex* d_dx_gpu;
         cuDoubleComplex* d_dy_gpu;
         cuDoubleComplex* d_dz_gpu;

         cudaMalloc((void**)&wave_gpu, nxyz * sizeof(cuDoubleComplex));
         cudaMalloc((void**)&fft_3_gpu, nxyz * sizeof(cuDoubleComplex));
         cudaMalloc((void**)&d_dx_gpu, nxyz * sizeof(cuDoubleComplex));
         cudaMalloc((void**)&d_dy_gpu, nxyz * sizeof(cuDoubleComplex));
         cudaMalloc((void**)&d_dz_gpu, nxyz * sizeof(cuDoubleComplex));

         // Create cuFFT plans
         cufftHandle forward_plan;
         cufftHandle backward_plan;
         cufftPlan3d(&forward_plan, nx, ny, nz, CUFFT_Z2Z);
         cufftPlan3d(&backward_plan, nx, ny, nz, CUFFT_Z2Z);

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

            wave[i] = make_cuDoubleComplex(cos(phase), sin(phase));
         }

         // for CUDA kernel to compute d/dx
         int blockSize = 256;
         int numBlocks = (nxyz + blockSize - 1) / blockSize;

         start = std::chrono::high_resolution_clock::now();
         // Copy data from CPU to GPU
         cudaMemcpy(wave_gpu, wave, nxyz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

         // Perform forward FFT
         cufftExecZ2Z(forward_plan, wave_gpu, fft_3_gpu, CUFFT_FORWARD);

         // make a copy of the forward transform
         cudaMemcpy(wave_gpu, fft_3_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

         // Compute d/dx
         computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, kx, nxyz);
         cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
         computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dx_gpu, static_cast<double>(nxyz),nxyz);

         // Compute d/dy
         computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, ky, nxyz);
         cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
         computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dy_gpu, static_cast<double>(nxyz),nxyz);

         // Compute d/dz
         computeDdx<<<numBlocks, blockSize>>>(wave_gpu, fft_3_gpu, kz, nxyz);
         cufftExecZ2Z(backward_plan, fft_3_gpu, fft_3_gpu, CUFFT_INVERSE);
         computeScale<<<numBlocks, blockSize>>>(fft_3_gpu, d_dz_gpu, static_cast<double>(nxyz),nxyz);

         // Copy data from GPU to CPU
         cudaMemcpy(d_dx, d_dx_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
         cudaMemcpy(d_dy, d_dy_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
         cudaMemcpy(d_dz, d_dz_gpu, nxyz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         
         elapsed_time = duration.count() * 1.e-9;

         long double performance = fp_op / 1.e6 / elapsed_time;
         trials.push_back(performance);

         // Free GPU memory
         cudaFree(wave_gpu);
         cudaFree(fft_3_gpu);
         cudaFree(d_dx_gpu);
         cudaFree(d_dy_gpu);
         cudaFree(d_dz_gpu);

         // Destroy cuFFT plans
         cufftDestroy(forward_plan);
         cufftDestroy(backward_plan);

         // Free CPU memory
         delete[] wave;
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

   std::ofstream file;
   file.open("cuFFTPerf.csv");
   for (int i=0; i<data.size(); i++) {
      file << data[i] << "," << std::endl;
   }
   file.close();

    return 0;
}
