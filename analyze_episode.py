import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = """
stats/condition_1 [0.0026 0.0024 0.0029 0.003  0.0023 0.0035 0.0046 0.0047 0.0056 0.0054
 0.2113 0.2141 0.1905 0.0074 0.1912 0.1899 0.199  0.0062 0.2127 0.0064
 0.0063 0.0071 0.0093 0.1746 0.0098 0.0149 0.009  0.1717 0.1789 0.0084
 0.1726 0.1712 0.1728 0.1726 0.165  0.0088 0.1702 0.1725 0.0134 0.1724
 0.0106 0.1699 0.1718 0.1783 0.1729 0.1702 0.0066 0.1682 0.1675 0.    ]
stats/condition_2 [0.0032 0.002  0.0023 0.0021 0.002  0.0029 0.006  0.0041 0.0067 0.0047
 0.1334 0.0098 0.0104 0.0069 0.126  0.012  0.0085 0.0088 0.1243 0.0064
 0.0067 0.0061 0.0051 0.1213 0.0072 0.0089 0.0086 0.1223 0.0126 0.0109
 0.1155 0.0113 0.0131 0.0099 0.0103 0.0069 0.1213 0.0208 0.0131 0.122
 0.0103 0.121  0.0131 0.0105 0.0061 0.006  0.0077 0.1147 0.0071 0.    ]
stats/condition_3 [0.002  0.0012 0.0012 0.0012 0.0012 0.0014 0.0023 0.0025 0.0024 0.0026
 0.1277 0.0079 0.007  0.0021 0.1339 0.01   0.0061 0.0047 0.1258 0.0025
 0.0042 0.0044 0.0034 0.1224 0.0023 0.0096 0.0052 0.1196 0.0072 0.0019
 0.1166 0.0077 0.0086 0.0076 0.0072 0.0027 0.1196 0.0092 0.0063 0.1203
 0.0023 0.1182 0.0095 0.0088 0.0036 0.0037 0.0047 0.1144 0.0048 0.    ]
stats/stages [0 0 0 0 0 0 0 0 0 0 2 3 3 0 2 3 3 0 2 0 0 0 0 2 0 0 0 2 3 0 2 3 3 3 3 0 2
 3 0 2 0 2 3 3 3 3 0 2 3 0]
"""

# Extract keys and arrays using regex
pattern = re.compile(r"stats/(\w+)\s+\[([^\]]+)\]", re.MULTILINE)
data_dict = {}

for match in pattern.finditer(data):
    key = match.group(1)
    array_str = match.group(2).strip().replace('\n', ' ')
    # Convert to list of floats or ints
    if key == "stages":
        arr = np.array(list(map(int, array_str.split())))
    else:
        arr = np.array(list(map(float, array_str.split())))
    data_dict[key] = arr

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# Save to CSV
df.to_csv("stats.csv", index=False)
print("CSV saved as stats.csv")

# Plot histograms
for key, values in data_dict.items():
    plt.figure()
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Histogram of {key}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
