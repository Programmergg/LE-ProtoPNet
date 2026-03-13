from math import pi
from os.path import join as pj
import matplotlib.pyplot as plt

# if your output looks like this:
# FINAL RESULTS:
# Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS
# 0.998   0.7353  0.602   0.532   0.54372 0.99826 0.54592 0.806
# 0.942   0.93719 0.964   0.942   0.32266 0.9949  0.8645  0.5
# Best threshold: 0.01620253164556962
# set results accordingly
# results = [0.998, 0.7353, 0.602, 0.532, 0.54372, 0.99826, 0.54592, 0.806] # SET YOUR VALUES HERE
results = [
    0.966,
    0.9479,
    0.962,
    0.948,
    0.91529,
    0.98145,
    0.72054,
    0.97879,
]  # SET YOUR VALUES HERE

def plot_results_funny_birds(results: list, save_path: str):
    acc = results[0]
    bi = results[5]
    com = ((results[1] + results[2] + results[3]) / 3 + results[4]) / 2  # ((CSDC + PC + DC) / 3 + Distractability) / 2
    cor = results[6]
    con = results[7]
    results = [acc, bi, com, cor, con]
    ax = plt.subplot(111, polar=True)
    categories = ["Acc.", "B.I.", "Com.", "Cor.", "Con."]
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    average = sum(results[2:]) / len(results[2:])
    results += results[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax.text(0, 0, (str(round(average, 2)) + "0")[1:4], horizontalalignment="center", verticalalignment="center", size=18)

    # Initialise the spider plot
    # ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels
    # ax.set_xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_xticks(angles[:-1], minor=False)
    ax.set_xticklabels(categories, fontdict=None, minor=False)
    ax.set_rlabel_position(36)
    ax.set_yticks([0.5, 1])
    ax.set_ylim(0, 1)
    color_string = "#555599"
    ax.plot(angles, results, linewidth=1, linestyle="solid", color=color_string)
    ax.fill(angles, results, color_string, alpha=0.2)
    plt.savefig(pj(save_path, "plot_funny_birds.png"), bbox_inches="tight", dpi=300)