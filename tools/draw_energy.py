import matplotlib.pyplot as plt
from datetime import datetime

# Provided data
data = """
Time    User  Nice   Sys  Idle    IO  Run Ctxt/s  IRQ/s Fork Exec Exit  Watts
01:10:55   0.5   0.0   0.7  98.8   0.0    1   1942   1006    0    0    0   1.43 
01:10:56   0.6   0.0   0.9  98.4   0.1    1   2003   1072    0    0    1   1.99 
01:10:57   0.8   0.0   0.5  98.8   0.0    1   2022   1053    0    0    0   1.65 
"""

# Parse the data into lists
lines = data.strip().split('\n')
time = []
watts = []
for line in lines[1:]:  # Skip the header line
    parts = line.split()
    time.append(datetime.strptime(parts[0], '%H:%M:%S'))
    watts.append(float(parts[-1]))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, watts, marker='o', linestyle='-')
plt.title('Time vs Watts')
plt.xlabel('Time')
plt.ylabel('Watts')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()
