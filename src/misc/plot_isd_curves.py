import numpy as np
import pandas as pd
import config as cfg
from sksurv.util import Surv
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times, make_time_bins
from utility.data import get_window_size, get_lag, get_lmd
from utility.event import EventManager
from sklearn.preprocessing import StandardScaler
from tools.regressors import BNNSurv
from utility.survival import Survival
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
import torch
from sklearn.model_selection import train_test_split
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from matplotlib.patches import Rectangle
import math

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
                     'font.size': 12.0,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

new_dataset = False
dataset = "xjtu"
n_boot = 0
dataset_path = cfg.DATASET_PATH_XJTU
n_bearing = cfg.N_REAL_BEARING_XJTU
bearing_ids = list(range(1, (n_bearing*2)+1))
pct_censoring = 0.25
n_post_samples = 100
bearing_axis = "X"

if __name__ == "__main__":
    data_util = DataETL(dataset, n_boot)
    event_manager = EventManager(dataset)
    fig, axes = plt.subplots(5, 3, figsize=(12, 16), sharey=True, layout='constrained')
    
    x_bearings = [idx for idx in list(range(0, 10)) if idx % 2 == 0]
    y_bearings = [idx for idx in list(range(0, 10)) if idx % 2 != 0]
    
    if new_dataset == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in [0, 1, 2]:
        timeseries_data, frequency_data = FileReader(dataset, dataset_path).read_data(test_condition, n_boot)
        event_times = EventManager(dataset).get_event_times(frequency_data, test_condition, lmd=get_lmd(test_condition))
        
        # Perform ISD on X/Y bearings
        if bearing_axis == "X":
            bearings = x_bearings
        else:
            bearings = y_bearings
            
        plot_indicies = [0, 1, 2, 3, 4]
        for test_idx, plot_idx in zip(bearings, plot_indicies):
            train_idx = [x for x in bearings if x != test_idx]
            train_data = pd.DataFrame()
            
            # Load train data
            for idx in train_idx:
                event_time = event_times[idx]
                transformed_data = data_util.make_moving_average(timeseries_data, event_time, idx+1,
                                                                 get_window_size(test_condition),
                                                                 get_lag(test_condition))
                train_data = pd.concat([train_data, transformed_data], axis=0)
            
            # Load test data
            test_event_time = event_times[test_idx]
            test_data = data_util.make_moving_average(timeseries_data, test_event_time, test_idx+1,
                                                      get_window_size(test_condition),
                                                      get_lag(test_condition))
            
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            unused_features = cfg.FREQUENCY_FTS + cfg.NOISE_FT
            train_data = train_data.drop(unused_features, axis=1)
            test_data = test_data.drop(unused_features, axis=1)
            train_data = Formatter.add_random_censoring(train_data, pct=pct_censoring)
            train_data = train_data.sample(frac=1, random_state=0)
            test_data = test_data.sample(frac=1, random_state=0)
            
            # Select only first observation
            test_sample = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()] \
                        .drop_duplicates(subset="Survival_time")
            print(test_sample)

            X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            X_test = test_sample.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_sample)
            
            # Set event times for models
            event_horizon = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            event_horizon = np.unique(event_horizon)
            
            # Scale data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #Format data
            t_train = y_train['Survival_time']
            e_train = y_train['Event']
            t_test = y_test['Survival_time']
            e_test = y_test['Event']
            
            #Set up the models on test
            model = BNNSurv().make_model(BNNSurv().get_best_hyperparams())

            # Train the model
            model.fit(X_train_scaled, t_train, e_train)
            
            # Predict
            surv_probs = model.predict_survival(X_test_scaled, event_horizon, n_post_samples)
            median_outputs = pd.DataFrame(np.mean(surv_probs, axis=0), columns=event_horizon)
            
            # Calculate TTE
            lifelines_eval = LifelinesEvaluator(median_outputs.T, t_test, e_test, t_train, e_train)
            pred_survival_time = int(lifelines_eval.predict_time_from_curve(predict_median_survival_time))
            print(pred_survival_time)
            
            # Plot
            p1 = axes[plot_idx, test_condition].plot(np.mean(median_outputs, axis=0).T, linewidth=2, label=r"$\mathbb{E}[S(t|\bm{X})]$", color="black")
            drop_num = math.floor(0.5 * n_post_samples * (1 - 0.9))
            lower_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=1+drop_num, dim=0)[0]
            upper_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=n_post_samples-drop_num, dim=0)[0]
            axes[plot_idx, test_condition].fill_between(event_horizon, upper_outputs[0,:], lower_outputs[0,:], color="gray", alpha=0.25)
            p2 = axes[plot_idx, test_condition].axhline(y=0.5, linestyle= "dashed", color='blue', linewidth=1, label='$\hat{y}_{i}$ = ' + f'{pred_survival_time}')        
            p3 = axes[plot_idx, test_condition].axvline(x=test_event_time, linestyle= "dashed",
                                                        color='green', linewidth=2.0, label=f'$y_i$ = {int(test_event_time)}')
            axes[plot_idx, test_condition].axvline(x=int(pred_survival_time), linestyle= "dashed", color='blue', linewidth=2.0)
            axes[plot_idx, test_condition].set_title(f'Bearing {test_condition+1}_{plot_idx+1}_{bearing_axis}')
            axes[plot_idx, test_condition].set_xlabel("Time (min)")
            text = f'Error = {int(pred_survival_time-test_event_time)}'
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            axes[plot_idx, test_condition].legend([p1[0], p2, p3, extra], [p1[0].get_label(), p2.get_label(), p3.get_label(), text], loc='upper right')
            if test_condition == 0:
                axes[plot_idx, test_condition].set_ylabel("Survival probability S(t)")
            axes[plot_idx, test_condition].grid(True)
    plt.savefig(f'{cfg.PLOTS_PATH}/individual_survival_axis_{bearing_axis.lower()}.pdf', format='pdf', bbox_inches="tight")