import numpy as np
import matplotlib.pyplot as plt
import csv

with open('cudaBlasPerf.csv', newline='') as f:
   reader = csv.reader(f)
   cuda = []
   for row in reader:
      cuda.append(float(row[0]))
   f.close()
   
with open('openBlasPerf.csv', newline='') as f:
   reader = csv.reader(f)
   openBlas = []
   for row in reader:
      openBlas.append(float(row[0]))
   f.close()
    
cuda_array = np.array(cuda)
openBlas_array = np.array(openBlas)
xvals = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
plt.loglog(xvals, cuda_array, "red")
plt.loglog(xvals, openBlas_array, "blue")
plt.title("BLAS Performance for cuBLAS and openBLAS")
plt.xlabel("Matrix Dimension")
plt.ylabel("Speed in mflops")
plt.legend(["cuBlas", "openBLAS"])
plt.show()