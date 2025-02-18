import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
# plt.rcParams["lines.linewidth"] = 25

plt.rcParams["axes.titlesize"] = 30
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class read_optuna_study(object):

    def __init__(self, study_name, result_form, model_name) -> None:

        storage_name = "sqlite:///{}.db".format(study_name)

        print(storage_name)

        if study_name == 0:
            self.study = 0
        else:
            self.study = optuna.load_study(study_name=study_name, storage=storage_name)

        self.result = result_form
        self.pareto_result = result_form
        self.nrow_all = 0
        self.nrow_pareto = 0
        self.model_name = model_name

    def read_all_result(self, data_src, target_label):

        for trial in self.study.trials:

            if not trial.values == None:
                for index, value in enumerate(list(trial.values)):
                    self.result["obj%d" % (index+1)].append(value)
                    if index == 0:
                        result_array = np.array(self.result["obj%d" % (index+1)]).reshape(-1,1)
                    else:
                        result_array = np.hstack([result_array, np.array(self.result["obj%d" % (index+1)]).reshape(-1,1)])
                    # print(index, ", ", value)

                for key, value in trial.params.items():
                    self.result[key].append(value)
                    # print(key, ", ", value)

        self.nrow_all, ncol = result_array.shape

        df = pd.DataFrame(result_array, columns=target_label)
        path = os.path.join(data_src, 'all_result_%s.csv' % self.model_name)
        df.to_csv(path)

        print('all result has been sent to %s' % path)


    def read_pareto_result(self, data_src, target_label):

        for trial in self.study.best_trials:

            if not trial.values == None:
                for index, value in enumerate(list(trial.values)):
                    self.pareto_result["obj%d" % (index+1)].append(value)
                    if index == 0:
                        pareto_result_array = np.array(self.pareto_result["obj%d" % (index+1)]).reshape(-1,1)
                    else:
                        pareto_result_array = np.hstack([pareto_result_array, np.array(self.pareto_result["obj%d" % (index+1)]).reshape(-1,1)])

                for key, value in trial.params.items():
                    self.pareto_result[key].append(value)

        self.nrow_pareto, ncol = pareto_result_array.shape

        df = pd.DataFrame(pareto_result_array, columns=target_label)
        path = os.path.join(data_src, 'pareto_result_%s.csv' % self.model_name)
        df.to_csv(path)

        print('pareto result has been sent to %s' % path)
        # pareto_front = pd.DataFrame(pareto_result)
        # pareto_front.to_csv("../OutData/pareto_front_NiTiFe-ML.csv")
        

    # Visualize the Pareto front using a scatter plot
    def pareto_front_scatter(self, data_src, target_label):

        data_all = pd.read_csv(os.path.join(data_src, 'all_result_%s.csv' % self.model_name))
        data_pareto = pd.read_csv(os.path.join(data_src, 'pareto_result_%s.csv') % self.model_name)
        line_pareto = pd.read_csv(os.path.join(data_src, 'pareto_result_%s.csv') % self.model_name).iloc[self.nrow_all:self.nrow_pareto] 

        line_pareto = line_pareto.sort_values(target_label[0], ascending = True)

        # line_pareto = pd.read_csv("../OutData/line_plot_pareto.csv")

        X1 = data_pareto[target_label[0]].values.flatten() 
        Y1 = data_pareto[target_label[1]].values.flatten() 

        X2 = data_all[target_label[0]].values.flatten() 
        Y2 = data_all[target_label[1]].values.flatten()

        X_line = line_pareto[target_label[0]].values.flatten() 
        Y_line = line_pareto[target_label[1]].values.flatten() 

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        # plt.axis("square")
        # print(len(X1))
        # print(len(X2))

        ax.scatter(50227.295799245774, 2.0011918294375493,s=150, marker="*", color="r",zorder=8)

        font_style = {"weight": "bold"}
        plt.legend(["Composition: Ni:47.8 Ti:50.2 Fe:2.0"], loc="upper left",prop=font_style)
        ax.scatter(X2, Y2, s=20, marker="x",color="g",alpha=0.5)
        ax.scatter(X1[len(X2):], Y1[len(Y2):], s=30, edgecolors="orange",linewidth=1.6, marker="D", color="w",zorder=2)
        

        # ax.plot(X_line, Y_line, linewidth=1, markersize=7, marker='d', linestyle='-',color='b',markerfacecolor='blue', markerfacecoloralt='w', markeredgecolor='k',fillstyle='top',zorder=3)
        # ax.plot(X_line, Y_line, linewidth=3, linestyle='-',color='#FFA500',zorder=5)
        # ax.plot([25000,150000], [1, 6], color="#d90166", linewidth=2, zorder=0) 
        # ax.plot([30000,130000], [1, 6], color="b", linestyle='--',linewidth=2, zorder=0,alpha=0.2) 

        a = np.array([[1, 1],
                      [2, 2]])
        color_list = ["#01659F", "#8FCEE3"]
        my_cmap = LinearSegmentedColormap.from_list('rain', color_list)
        cm.register_cmap(cmap=my_cmap)
        ax.imshow(a, interpolation='bicubic', extent=(20000,130000, 0, 6), cmap=my_cmap,alpha=0.5)
        ax.set_aspect('auto')
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        y_lable = target_label[1] + " Fraction"

        plt.ylabel(y_lable,weight='bold')
        plt.xlabel(target_label[0],weight='bold')
        plt.yticks(weight='bold')
        plt.xticks(weight='bold')
        plt.ylim([0, 6])
        plt.xlim([20000,130000])

        plt.savefig("../OutPlot/ParetoOptimization.png",dpi=600)

if __name__ == "__main__":

    result_form =  {
            "obj1": [], "obj2": [],  "Ni": [], "Ti":[], "Fe":[]
        }
    study_name = "../model/NiTiFe_test_Catboost"
    data_src = '../Output_Data'
    target_label = ["CSI","Ti2Ni"]
    model_name = "Catboost"

    sample = read_optuna_study(study_name, result_form, model_name)
    sample.read_all_result(data_src, target_label)
    sample.read_pareto_result(data_src, target_label)
    sample.pareto_front_scatter(data_src, target_label)

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    fig = optuna.visualization.plot_pareto_front(study, target_names=["CSI", "Ti2Ni"])
    fig.show()


    