from matplotlib import pyplot as plt
import numpy as np
import math
import csv

results = open("results.csv", "r")

reader = csv.reader(results)

data = list(reader)
for xd in data:
    print(xd)


fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7,13))

naives = {}
comp = {}
decomp = {}

for row in data:
    acc, threads, naive_compress, compress, decompress = [float(item) for item in row]

    if threads not in naives:
        naives[threads] = []

    if threads not in comp:
        comp[threads] = []
    
    if threads not in decomp:
        decomp[threads] = []

    naives[threads].append((acc, naive_compress))
    comp[threads].append((acc, compress))
    decomp[threads].append((acc, decompress))

for (threads, info) in naives.items():
    info = np.array(info)
    ax1.plot(info[:,0], info[:,1], label=threads, marker="o", markersize=5)

for (threads, info) in comp.items():
    info = np.array(info)
    ax2.plot(info[:,0], info[:,1], label=threads, marker="o", markersize=5)

for (threads, info) in decomp.items():
    info = np.array(info)
    ax3.plot(info[:,0], info[:,1], label=threads, marker="o", markersize=5)


fig.suptitle("Relation of speedup to accuracy")
ax1.legend(title="Number of threads")
ax1.set_title("Naive compression")
ax2.set_title("Compression")
ax3.set_title("Decompression")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim([0., 3.])
    ax.set_xticks([8, 16, 32])

ax2.set_ylabel("Speedup")
ax3.set_xlabel("Accuracy")

plt.savefig("plot.png", bbox_inches='tight')
plt.show()
