import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 19
plt.rcParams["ytick.labelsize"] = 19
# plt.rcParams["lines.linewidth"] = 25

plt.rcParams["axes.titlesize"] = 30
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

dataset = pd.read_csv("../Dataset/model-CSI-error.csv")


labels = dataset["Model name"]  # the label locations
x = np.arange(len(labels))
ys = dataset["RMSE"]
state = dataset["state"]

width = 0.3  # the width of the bars
multiplier_1 = 0.5
multiplier_2 = -0.5

def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
     aa[i, i] = i
    return aa

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

plt.axis("equal")
# plt.axes(aspect='equal')

offset_1 = width * multiplier_1
offset_2 = width * multiplier_2 - 5


# ax.fill_between([-0.5,4.5], 0, 15000,color="#8FCEE3", alpha=0.5)

a = np.array([[1, 1],
              [2, 2]])
color_list = ["#01659F", "#8FCEE3"]

my_cmap = LinearSegmentedColormap.from_list('rain', color_list)

cm.register_cmap(cmap=my_cmap)

ax.imshow(a, interpolation='bicubic', extent=(-0.5, 4.5, 0, 15000), cmap=my_cmap,alpha=0.5)

ax=plt.gca()
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['top'].set_linewidth(4)



plt.bar(x[5:10] + offset_2, ys[5:10], width, color="#F8C6B5",  edgecolor='black',linewidth=3, label="Default Hyper Parameters")
plt.bar(x[:5] + offset_1, ys[:5], width, color="#A172D0", edgecolor='black',linewidth=3,label="Optimization Hyper Parameters")

# ax.fill_between(x[5:10] + offset_2, ys[5:10], y2=0,
#                             hatch='///', linewidth=3, zorder=2, fc='c')

# for index, label in enumerate(labels):

#     if state[index] == "Opti":
#         offset = width * multiplier_1
#         rects = ax.bar(x + offset, ys, width, label="Optimization Hyper Parameters")

#     if state[index] == "Default":
#         offset = width * multiplier_2
#         index_x = index - 5
#         rects = ax.bar(index_x + offset, ys[index], width, label="Default Hyper Parameters")
#         # ax.bar_label(rects, padding=2)
#         # multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
# plt.yticks(weight='bold')
# plt.xticks(weight='bold')
plt.ylabel('RMSE',weight='bold')
plt.xlabel('Model',weight='bold')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

plt.title('RMSE of Different Machine Learning Model',weight='bold')
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['KNN', 'LightGBM', 'Catboost', 'SVR', 'RF'])
ax.set_aspect('auto')
plt.legend(loc='upper left')
plt.xlim([-0.5, 4.5])
plt.ylim([0, 15000])

plt.savefig("../Outplot/RMSE of Different Model.png", dpi=600)