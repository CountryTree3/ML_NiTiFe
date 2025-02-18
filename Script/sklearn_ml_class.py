import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler as DataScaler
# from sklearn.preprocessing import RobustScaler as DataScaler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn import linear_model
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from pandas import MultiIndex, Int16Dtype
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import warnings
import matplotlib.ticker as mtick
warnings.filterwarnings("ignore")

plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
# plt.rcParams["lines.linewidth"] = 25

plt.rcParams["axes.titlesize"] = 30
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# class CalML, from sklearn.preprocessing import MinMaxScaler as DataScaler
class CalML(object):
   def __init__(self, src, feature_labels, target_labels, random_state) -> None:
    
      # read data
      self.dataset = pd.read_csv(src)
      self.X_sample = self.dataset[feature_labels].values
      self.Y_sample = self.dataset[target_labels].values

      self.feature_labels = feature_labels
      self.target_labels = target_labels

      # data preprocessing
      self.feauture_ss = DataScaler()
      self.target_tt = DataScaler()
      self.feauture_ss.fit(self.X_sample)
      self.target_tt.fit(self.Y_sample)

      self.xData_mm, self.yData_mm = self.feauture_ss.transform(self.X_sample), self.target_tt.transform(self.Y_sample) 
      self.xData = pd.DataFrame(self.xData_mm,columns=feature_labels)
      self.yData = pd.DataFrame(self.yData_mm,columns=target_labels) 

      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.xData, self.yData, test_size=0.2,random_state=random_state)

# train and test, then return the error
   def model_func(self, model_name, **param):

      # train
      if model_name == "RF":
         model = RandomForestRegressor(**param)
         model.fit(self.X_train,  np.ravel(self.y_train))  

      elif model_name == "Catboost":

         model = CatBoostRegressor(**param, iterations=1000)
         # model.fit(self.X_train.values, self.y_train.values, eval_set=[(self.X_test.values, self.y_test.values)], verbose=False, early_stopping_rounds=200)  
         model.fit(self.X_train.values, self.y_train.values, verbose=False)       

      elif model_name == "GPR":
         best= {'constant_value':0 , 'noise':0, 'nu':0 , 'scaler':0 }
         noise = best["noise"]
         m52 = ConstantKernel(best['constant_value']) * Matern(length_scale=best["scaler"], nu=best["nu"])
         model = GaussianProcessRegressor(kernel=m52, alpha=noise**2)
         model.fit(self.X_train,  self.y_train)

      elif model_name == "KNN":
         model = KNeighborsRegressor(**param)
         model.fit(self.X_train, self.y_train)  

      elif model_name == "SVR":
         model = SVR(**param)
         model.fit(self.X_train.values,  np.ravel(self.y_train.values)) 

      elif model_name == "XGBoost":
         model = XGBRegressor(**param)
         model.fit(self.X_train.values, self.y_train.values, eval_set=[(self.X_test.values, self.y_test.values)], verbose=False, early_stopping_rounds=500)   

      elif model_name == "LightGBM":
         if os.path.exists("../model/model_%s_%s opti.pkl" %("CSI", model_name)): 
            model = joblib.load("../model/model_%s_%s opti.pkl" %("CSI", model_name))
            model.fit(self.X_train.values, self.y_train.values, eval_set=[(self.X_test.values, self.y_test.values)], verbose=False, early_stopping_rounds=500)       
         else:
            model = LGBMRegressor(**param)
            model.fit(self.X_train.values, self.y_train.values, verbose=False)  

      else:
         print("No this model name, Please Check the Variable 'model_name' !!!")  
         exit(0)   

      # predict
      y_predict_test = self.target_tt.inverse_transform(model.predict(self.X_test).reshape(-1,1))
      y_predict_train = self.target_tt.inverse_transform(model.predict(self.X_train).reshape(-1,1))

      self.y_test_init = self.target_tt.inverse_transform(self.y_test.values).reshape(-1,1)
      self.y_train_init = self.target_tt.inverse_transform(self.y_train.values).reshape(-1,1)

      # evaluation
      self.RMSE_test = np.sqrt(mean_squared_error(self.y_test_init, y_predict_test))
      RMSE_train = np.sqrt(mean_squared_error(self.y_train_init, y_predict_train))
      
      self.R2_test = r2_score(self.y_test_init, y_predict_test)
      R2_train = r2_score(self.y_train_init, y_predict_train) 

      return self.RMSE_test, RMSE_train, self.R2_test, R2_train, y_predict_test, y_predict_train, model

# visualization
   def plot_output(self, model_name, target_labels, range, train_pred, test_pred):

      range = np.array(range)

      fig = plt.figure(figsize=(9,8))
      ax = fig.add_subplot(111)
      plt.axis("square")   

     # background filled
      plt.scatter(self.y_train_init, train_pred, s=50,color="green",marker="D",alpha=0.8,edgecolors="black",linewidth=1.5,zorder=2)
      plt.scatter(self.y_test_init, test_pred, s=100, color="#a55af4", edgecolors="black",linewidth=1.5, zorder=5)  # "#DC143C"是红色
      plt.legend(["Training Data","Testing Data"])


      # CSI
      # plt.scatter(self.y_test_init, test_pred, s=100, color="#a55af4", edgecolors="black",linewidth=1.5, zorder=5)  # "#DC143C"是红色
      # plt.legend(["Testing Data"])      
 
      a = np.array([[1, 1],
                    [2, 2]])
      color_list = ["#01659F", "#8FCEE3"]
      my_cmap = LinearSegmentedColormap.from_list('rain', color_list)
      cm.register_cmap(cmap=my_cmap)
      ax.imshow(a, interpolation='bicubic', extent=(np.min(range), np.max(range), np.min(range), np.max(range)), cmap=my_cmap,alpha=0.5)

      # diagonal line
      plt.plot(range, range, color="#d90166", linewidth=3, zorder=4) 
      # plt.plot(range, range, color="k", linewidth=3, zorder=4) 

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
      # plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')

      # ax.set_aspect('auto')
      if not bool(param):
         plt.title("%s - Default" % (model_name), weight='bold')
         plt.savefig("../Outplot/%s predict %s default params.png" % (model_name, target_labels), dpi=300)

      else:
         plt.title("%s - Opti" % (model_name), weight='bold')
         plt.savefig("../Outplot/%s predict %s opti params.png" % (model_name, target_labels), dpi=300)
      
      # plt.savefig("../Outplot/%s predict %s.png" % (model_name, target_labels), dpi=300)


if __name__ == '__main__':
   # ——————Initialize dataset——————
   independent_vars_labels = ["Ni", "Ti", "Fe"]
   dependent_vars_labels = ["CSI"]
   src = "../Dataset/NiTiFe-ML.csv"

   # ——————Choose model and hyper params——————
   value_range = [30000, 150000]
   # value_range = [0, 25]

   # available model name: Catboost, KNN, SVR, GPR(certain kenerl function), RF, XGBboost, LightGBM
   model_name = "Catboost"
   predict = "off"
   param = {}

   # The following are not optimal hyperparameters of Catboost model, 
   # this only displays the hyperparameters that have be optimized
   param = {   
      "boosting_type": "Plain",
      "bootstrap_type": "Bernoulli",
      "colsample_bylevel": 0.1374563844640812,
      "depth": 10,
      "l2_leaf_reg": 3.768449678015936,
      "learning_rate": 0.08129221226818754,
      "loss_function": "Poisson",
      "min_data_in_leaf": 9,
      "one_hot_max_size": 8,
    }
   random_state = 68
 
   for y_label in dependent_vars_labels:
      print("##### dependent_vars_labels is %s #####" % y_label)

      # ——————train and test, output error
      ML_calc = CalML(src, independent_vars_labels, [y_label], random_state)
      RMSE_test, RMSE_train, R2_test, R2_train, y_predict_test, y_predict_train, model = ML_calc.model_func(model_name, **param)

      print("when target is %s, RMSE_test = %.4f" %(y_label, RMSE_test))
      print("when target is %s, R2_test = %.4f" %(y_label, R2_test))

      # ——————output predict result of test dataset to csv
      output = pd.DataFrame({  "true":ML_calc.y_test_init.flatten(),
                              "predict":y_predict_test.flatten(),                        
                               })
      output.to_csv("../OutData/predict_%s_%s.csv" % (y_label, model_name))

      # ——————output pictrue of test predict result
      ML_calc.plot_output(model_name, y_label, value_range, y_predict_train, y_predict_test)

      # ——————save the model to specific folder
      if not bool(param):
         joblib_file = "../model/model_%s_%s default.pkl" % (y_label, model_name)
         joblib.dump(model, joblib_file)
      else:
         joblib_file = "../model/model_%s_%s opti.pkl" % (y_label, model_name)
         joblib.dump(model, joblib_file)
   


