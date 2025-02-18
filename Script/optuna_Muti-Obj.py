import logging
import sys
import optuna
import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn_ml_class import CalML
import joblib
import pandas as pd
from optuna.samplers import NSGAIISampler



def new_study(study_name, model_name):

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=NSGAIISampler(seed=seed), directions=["minimize", "minimize"])

    def objective(trial):
        
        Ti = trial.suggest_float("Ti", 48, 55, step=0.02)
        Fe = trial.suggest_float("Fe", 0, 4, step=0.02)       
        Ni = 100 - Ti - Fe

        Composition = np.array([Ni, Ti, Fe])
        # print(Composition)

        result = ML_predict(Composition.reshape(1, -1), model_name, ['CSI','Ti2Ni'])

        result_1 = result[0, 0]
        result_2 = result[0, 1]

        return result_1, result_2

    study.optimize(objective, n_trials=n_trials)

    fig = optuna.visualization.plot_pareto_front(study, target_names=["CSI", "Ti2Ni"])
    fig.show()

    # print("Best: ", study.best_trials)

def load_study(study_name, model_name):
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage_name, sampler=NSGAIISampler(seed=seed))

    def objective(trial):

        # Ni = trial.suggest_float("Ni", 40, 50)
        Ti = trial.suggest_float("Ti", 48, 55, step=0.02)
        Fe = trial.suggest_float("Fe", 0, 4, step=0.02)       
        Ni = 100 - Ti - Fe

        Composition = np.array([Ni, Ti, Fe])
        # print(Composition)

        result = ML_predict(Composition.reshape(1, -1), model_name, ['CSI','Ti2Ni'])

        result_1 = result[0, 0]
        result_2 = result[0, 1]


        return result_1, result_2

    study.optimize(objective, n_trials=n_trials)
    fig = optuna.visualization.plot_pareto_front(study, target_names=["CSI", "Ti2Ni"])
    fig.show()
    # fig.savefig("../Outplot/Ti2Ni and CSI Pareto Front.png", dpi=600)
    
# def read_and_init(independent_vars_labels, dependent_vars_labels, src):
    

def ML_predict(X, model_name, target):

    independent_vars_labels = ["Ni", "Ti", "Fe"]
    dependent_vars_labels = ["CSI", "Ti2Ni"]
    src = "../Dataset/NiTiFe-ML.csv"

    feauture_ss = CalML(src, independent_vars_labels, dependent_vars_labels).feauture_ss
    target_tt = CalML(src, independent_vars_labels, dependent_vars_labels).target_tt
    Composition_mm = feauture_ss.transform(X.reshape(1,-1))

    Comp_mm_Dataframe = pd.DataFrame(Composition_mm,columns=independent_vars_labels)
    # print(Comp_mm_Dataframe)

    y = np.zeros(len(target))

    for index, label in enumerate(target):
    
      model = joblib.load("../model/model_%s_%s opti.pkl" %(label, model_name))
        
      y[index] = model.predict(Comp_mm_Dataframe)
    
    result = target_tt.inverse_transform(y.reshape(1,-1))
    result = y.reshape(1,-1)

    return result

if __name__ == "__main__":

    model_name = "Catboost"

    # seed = 66
    seed = 283

    n_trials = 500

    if os.path.exists('../model/NiTiFe_test_%s_seed%d.db' % (model_name, seed)):
        load_study("../model/NiTiFe_test_%s_seed%d" % (model_name, seed), model_name )

    else:
        new_study("../model/NiTiFe_test_%s_seed%d" % (model_name, seed), model_name )
    






























exit(0)




# 下面是用pymoo写的
import numpy as np
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.termination import get_termination
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import joblib
from sklearn_ml_class import CalML

class MyProblem(ElementwiseProblem):
 
    def __init__(self, model_CSI, model_Ti2Ni, target_ss_CSI, target_ss_Ti2Ni):

        self.model_CSI = model_CSI
        self.model_Ti2Ni = model_Ti2Ni
        self.target_ss_CSI = target_ss_CSI
        self.target_ss_Ti2Ni = target_ss_Ti2Ni

        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=1,
                         # Ti, Fe, Ni
                         xl=np.array([50, 0, 40]),
                         xu=np.array([55, 5, 50]))
 
    def _evaluate(self, x, out, *args, **kwargs):

        # f1 = self.target_ss_CSI.inverse_transform(self.model_CSI.predict(x.reshape(1,-1)).reshape(-1,1))
        # f2 = self.target_ss_Ti2Ni.inverse_transform(self.model_Ti2Ni.predict(x.reshape(1,-1)).reshape(-1,1))
        f1 = self.model_CSI.predict(x.reshape(1,-1))
        f2 = self.model_Ti2Ni.predict(x.reshape(1,-1))

        g1 = x[0]+x[1]+x[2]-100
        # g2 = 1-x[1]

        out["F"] = [f1, f2]
        out["G"] = [g1]


independent_vars_labels = ["Ti", "Fe", "Ni"]
dependent_vars_CSI = ["CSI"]
dependent_vars_Ti2Ni = ["Ti2Ni"]
src = "../Dataset/NiTiFe-ML.csv"

target_ss_CSI = CalML(src, independent_vars_labels, dependent_vars_CSI).target_tt
target_ss_Ti2Ni = CalML(src, independent_vars_labels, dependent_vars_Ti2Ni).target_tt

model_csi = joblib.load("../model/model_CSI.pkl")
model_ti2ni = joblib.load("../model/model_ti2ni.pkl")

problem = MyProblem(model_csi, model_ti2ni, target_ss_CSI, target_ss_Ti2Ni)

 
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
 
termination = get_termination("n_gen", 40)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
X = res.X
F = res.F
 
 

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='red')
plt.title("Objective Space")
plt.show()
