import numpy as np
import matplotlib.pyplot as plt
import csv

with open('cudaHostToDeviceSpeed.csv', newline='') as f:
   reader = csv.reader(f)
   HostToDevice = []
   for row in reader:
      HostToDevice.append(float(row[0]))
   f.close()
   
with open('cudaDeviceToHostSpeed.csv', newline='') as f:
   reader = csv.reader(f)
   DeviceToHost = []
   for row in reader:
      DeviceToHost.append(float(row[0]))
   f.close()
    
HostToDeviceTimes = np.array(HostToDevice)
DeviceToHostTimes = np.array(DeviceToHost)

xvalslist = []
i = 8
while i <= 268435456:
   xvalslist.append(i)
   i *= 2
xvals = np.array(xvalslist)
plt.semilogx(xvals, HostToDeviceTimes, "red")
plt.semilogx(xvals, DeviceToHostTimes, "blue")
plt.title("CPU to GPU Data Copy Speed")
plt.xlabel("Buffer Size in Bytes")
plt.ylabel("Bandwidth in Bytes per Second")
plt.legend(["Host to Device", "Device to Host"])
plt.show()