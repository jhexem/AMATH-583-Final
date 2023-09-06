import numpy as np
import matplotlib.pyplot as plt
import csv

with open('rowSwapTimesFile.csv', newline='') as f:
   reader = csv.reader(f)
   rswap = []
   for row in reader:
      rswap.append(float(row[0]))
   f.close()
   
with open('colSwapTimesFile.csv', newline='') as f:
   reader = csv.reader(f)
   cswap = []
   for row in reader:
      cswap.append(float(row[0]))
   f.close()
    
row_times = np.array(rswap)
col_times = np.array(cswap)
xvals = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
plt.loglog(xvals, row_times, "red")
plt.loglog(xvals, col_times, "blue")
plt.title("Swap Times for Rows and Columns in File")
plt.xlabel("Matrix Dimension")
plt.ylabel("Time in seconds")
plt.legend(["rows", "columns"])
plt.show()