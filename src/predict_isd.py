import numpy as np
import pandas as pd
import math
import torch
import config as cfg
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.regressors import CoxPH, RSF, DeepSurv, DSM, BNNmcd
from tools.feature_selectors import PHSelector
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines.statistics import proportional_hazard_test
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
import config as cfg
from sklearn.preprocessing import StandardScaler
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times
from utility.data import get_window_size, get_lag

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
                     'font.size': 12.0,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

from matplotlib.patches import Rectangle

DATASET = "xjtu"
N_BOOT = 0
N_POST_SAMPLES = 1000
DATASET_PATH = cfg.DATASET_PATH_XJTU

if __name__ == "__main__":

    data_util = DataETL(DATASET, N_BOOT)
    fig, axes = plt.subplots(5, 3, figsize=(12, 16), sharey=True)

    x_bearings = [idx for idx in list(range(1, 11)) if idx % 2 != 0]
    y_bearings = [idx for idx in list(range(1, 11)) if idx % 2 == 0]

    for cond in [0, 1, 2]:
        # Build timeseries data
        timeseries_data, boot, info_pack = FileReader(DATASET, DATASET_PATH).read_data(cond, N_BOOT)
        window_size = get_window_size(cond)
        lag = get_lag(cond)
        
        # Individual bearing prediction
        bearings = y_bearings
        plot_indicies = [0, 1, 2, 3, 4]
        for test_idx, plot_idx in zip(bearings, plot_indicies):
            train_idx = [x for x in bearings if x != test_idx]
            
            train_data = pd.DataFrame()
            for idx in train_idx:
                event_time = data_util.event_analyzer(idx, info_pack)
                transformed_data = data_util.make_moving_average(timeseries_data, event_time, idx, window_size, lag)
                train_data = pd.concat([train_data, transformed_data], axis=0)
                
            test_event_time = data_util.event_analyzer(test_idx, info_pack)
            test_data = data_util.make_moving_average(timeseries_data, test_event_time, test_idx, window_size, lag)
        
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            
            train_data = Formatter.add_random_censoring(train_data, percentage=0.25)
            test_data = Formatter.add_random_censoring(test_data, percentage=0.25)
            train_data = train_data.sample(frac=1, random_state=0)
            test_data = test_data.sample(frac=1, random_state=0)
            
            # Select only first observation
            test_sample = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()] \
                          .drop_duplicates(subset="Survival_time")

            x_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            x_test = test_sample.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_sample)

            #Set event times for models
            event_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            event_times = np.unique(event_times)

            #Set the feature selector and train/test split
            best_features = PHSelector(x_train, y_train, estimator=[DATASET, cond]).get_features()
            X_train, X_test = x_train.loc[:,best_features], x_test.loc[:,best_features]

            # Scale data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #Format the data
            t_train = y_train['Survival_time']
            e_train = y_train['Event']
            t_test = y_test['Survival_time']
            e_test = y_test['Event']
            
            # Set event time
            failure_time = data_util.event_analyzer(test_idx, info_pack)
            
            #Set up the models on test
            model = BNNmcd().make_model(BNNmcd().get_best_hyperparams())

            # Train the model
            model.fit(X_train_scaled, t_train, e_train)
            
            # Predict
            surv_probs = model.predict_survival(X_test_scaled, event_times, N_POST_SAMPLES)
            median_outputs = pd.DataFrame(np.mean(surv_probs, axis=0), columns=event_times)
            
            # Calculate TTE
            lifelines_eval = LifelinesEvaluator(median_outputs.T, t_test, e_test, t_train, e_train)
            median_survival_time = round(np.median(lifelines_eval.predict_time_from_curve(predict_median_survival_time)))
            
            # Plot
            p1 = axes[plot_idx, cond].plot(np.mean(median_outputs, axis=0).T, linewidth=2, label=r"$\mathbb{E}[S(t|\bm{X})]$", color="black")
            drop_num = math.floor(0.5 * N_POST_SAMPLES * (1 - 0.9))
            lower_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=1+drop_num, dim=0)[0]
            upper_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=N_POST_SAMPLES-drop_num, dim=0)[0]
            axes[plot_idx, cond].fill_between(event_times, upper_outputs[0,:], lower_outputs[0,:], color="gray", alpha=0.25)
            p2 = axes[plot_idx, cond].axhline(y=0.5, linestyle= "dashed", color='blue', linewidth=1, label='$\hat{y}_{i}$ = ' + f'{median_survival_time}')        
            p3 = axes[plot_idx, cond].axvline(x=test_event_time, linestyle= "dashed",
                                              color='green', linewidth=2.0, label=f'$y_i$ = {int(test_event_time)}')
            axes[plot_idx, cond].axvline(x=int(median_survival_time), linestyle= "dashed", color='blue', linewidth=2.0)
            axes[plot_idx, cond].set_title(f'Bearing {cond+1}_{plot_idx+1}_Y')
            axes[plot_idx, cond].set_xlabel("Time (min)")
            text = f'Error = {int(median_survival_time-test_event_time)}'
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            axes[plot_idx, cond].legend([p1[0], p2, p3, extra], [p1[0].get_label(), p2.get_label(), p3.get_label(), text], loc='upper right')
            if cond == 0:
                axes[plot_idx, cond].set_ylabel("Survival probability S(t)")
            axes[plot_idx, cond].grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.PLOTS_PATH}/individual_survival_axis_y.pdf', format='pdf', bbox_inches="tight")