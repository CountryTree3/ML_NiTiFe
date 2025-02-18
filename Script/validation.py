import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as DataScaler
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from sklearn_ml_class import CalML
warnings.filterwarnings("ignore")

def validation_plot(model, exp_x, exp_y):

      pred_y = model(exp_x)    

      range = np.array(range)

      exp_x_norm = model.feauture_ss.transform(exp_x)

      pred_y = model.target_tt.inverse_transform(model.predict(exp_x_norm).reshape(-1,1))

      fig = plt.figure(figsize=(9,8))
      ax = fig.add_subplot(111)
      plt.axis("square")   

      # background filled
      plt.scatter(exp_y, pred_y, s=50,color="green",marker="D",alpha=0.8,edgecolors="black",linewidth=1.5,zorder=2)
      plt.legend(["Training Data","Testing Data"])
      a = np.array([[1, 1],
                    [2, 2]])
      color_list = ["#01659F", "#8FCEE3"]
      my_cmap = LinearSegmentedColormap.from_list('rain', color_list)
      cm.register_cmap(cmap=my_cmap)
      ax.imshow(a, interpolation='bicubic', extent=(np.min(range), np.max(range), np.min(range), np.max(range)), cmap=my_cmap,alpha=0.5)
            
      # diagonal line
      plt.plot(range, range, color="#d90166", linewidth=3, zorder=4) 

      ax=plt.gca()

      ax.spines['bottom'].set_linewidth(4)
      ax.spines['left'].set_linewidth(4)
      ax.spines['right'].set_linewidth(4)
      ax.spines['top'].set_linewidth(4)


      plt.ylim([np.min(range), np.max(range)])
      plt.xlim([np.min(range), np.max(range)])
      plt.yticks(weight='bold')
      plt.xticks(weight='bold')
      plt.ylabel("Predicted",weight='bold')
      plt.xlabel("Calculated",weight='bold')    

      plt.rcParams["axes.labelsize"] = 20
      plt.rcParams['font.weight'] = "bold"
      plt.rcParams['axes.labelweight'] = "bold"

      plt.savefig("../OutPlot/NiTiFe-Density-valid.png", dpi=600, transparent=True)
      # plt.show()
      plt.clf()
      plt.close()


if __name__ == '__main__':

    independent_vars_labels = [""]
    dependent_vars_labels = [""]
    src = ""

    model = CalML(independent_vars_labels, dependent_vars_labels, src)


    param = {   
    }
      
    exp_x = pd.DataFrame({
      })

    validation_plot()