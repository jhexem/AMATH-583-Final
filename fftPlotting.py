import numpy as np
import matplotlib.pyplot as plt
import csv

with open('cuFFTPerf.csv', newline='') as f:
   reader = csv.reader(f)
   cuda = []
   for row in reader:
      cuda.append(float(row[0]))
   f.close()

with open('fftwPerf.csv', newline='') as f:
   reader = csv.reader(f)
   fftw = []
   for row in reader:
      fftw.append(float(row[0]))
   f.close()

cuda_array = np.array(cuda)
fftw_array = np.array(fftw)
xvals = np.array([16, 32, 64, 128, 256, 512])
plt.loglog(xvals, cuda_array, "red")
plt.loglog(xvals, fftw_array, "blue")
plt.title("FFT Performance for cuFFT and FFTW")
plt.xlabel("Lattice Size")
plt.ylabel("Speed in mflops")
plt.legend(["cuFFT", "FFTW"])
plt.show()