import numpy as np
import pandas as pd

class AlarmLabeler:

    def __init__(self):
        pass

    def get_alarms(self, data, citizen_ids, date_dict):
        alarms = np.zeros(len(citizen_ids))
        for i, citizen_id in enumerate(citizen_ids):
            citizen_alarm = np.where(data == citizen_id)[0]
            if citizen_alarm.size == 0:
                alarms[i] = np.inf
            else:
                idx = citizen_alarm[0]
                year, week = data[idx][1], data[idx][2]
                if year == '' or week == '':
                    alarms[i] = np.inf
                else:
                    key = (int(year), int(week))
                    if key in date_dict:
                        ts = date_dict[(int(year), int(week))]
                        alarms[i] = ts
                    else:
                        alarms[i] = np.inf
        return alarms

    def get_dropouts(self, data, start_at_ts, citizen_ids, dropout_threshold):
        sum_of_care = np.sum(data[:,:,2:], axis=2)
        zero_ranges = np.array(list(map(self.zero_runs, sum_of_care)), dtype=object)
        dropouts = np.full(len(citizen_ids), np.inf)
        for i, zero_range in enumerate(zero_ranges):
            for pair in zero_range:
                if pair[0] > start_at_ts[i]:
                    if (pair[1] - pair[0]) >= dropout_threshold:
                        dropouts[i] = pair[0]
                        continue
        return dropouts

    def get_starts(self, data, ts_len):
        sum_of_care = np.sum(data, axis=2)
        starts = np.full(len(sum_of_care), ts_len)
        for i in range(len(sum_of_care)):
            idx = np.where(sum_of_care[i]>0)
            starts[i] = idx[0][0]
        return starts

    def zero_runs(self, a):
        # https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    def get_first_event(self, alarm_ts: float, dropout_ts: float) -> str:
        min_idx = np.argmin([alarm_ts, dropout_ts])
        if min_idx == 0:
            if alarm_ts > 0 and alarm_ts < np.inf:
                return "Alarm"
            return "NoEvent"
        elif min_idx == 1:
            return "Dropout"

    def make_alarm_label(self, df: pd.DataFrame, start_at_ts: np.ndarray,
                        alarm_at_ts: np.ndarray, dropout_at_ts: np.ndarray) -> pd.DataFrame:
        """
        Takes a dataframe and assigns remaining weeks until citizen gets an
        alarm, drops out or study ends. The variable Observed represents whether
        the corresponding event was observed (1) or censored (0). 
        Alarms/Falls are observed and Dropout/NoEvent are censoired.
        The variabels Event, Dropout and Started are auxiliary variables.
        """
        label_cols = ['Weeks', 'Observed']
        aux_cols = ['Event', 'Started']
        df = df.reindex(columns=df.columns.tolist() + label_cols + aux_cols).fillna(0)
        citizen_ids = df['Id'].unique()
        for citizen_id in citizen_ids:
            obs_length = len(df.loc[df['Id'] == citizen_id])
            start_ts, alarm_ts, dropout_ts = start_at_ts[citizen_id], alarm_at_ts[citizen_id], dropout_at_ts[citizen_id]
            df.loc[df['Id'] == citizen_id, 'Started'] = [0 if x < int(start_ts) else 1 for x in range(obs_length)]
            first_event = self.get_first_event(alarm_ts, dropout_ts)
            if first_event == "Alarm":
                df.loc[df['Id'] == citizen_id, 'Event'] = [0 if x < int(alarm_ts) else 1 for x in range(obs_length)]
                event = df.groupby((df.loc[df['Id'] == citizen_id, 'Event'] == 1).cumsum())
                df.loc[df['Id'] == citizen_id, 'Weeks'] = event.cumcount(ascending=False)+1
                df.loc[df['Id'] == citizen_id, 'Observed'] = 1
            elif first_event == "Dropout":
                df.loc[df['Id'] == citizen_id, 'Event'] = [0 if x < int(dropout_ts) else 1 for x in range(obs_length)]
                event = df.groupby((df.loc[df['Id'] == citizen_id, 'Event'] == 1).cumsum())
                df.loc[df['Id'] == citizen_id, 'Weeks'] = event.cumcount(ascending=False)+1
                df.loc[df['Id'] == citizen_id, 'Observed'] = 0
            elif first_event == "NoEvent":
                df.loc[df['Id'] == citizen_id, 'Weeks'] = list(df.loc[df['Id'] == citizen_id, 'Time'].iloc[::-1]+1)
                df.loc[df['Id'] == citizen_id, 'Observed'] = 0
            if start_ts > 0:
                df.loc[df['Id'] == citizen_id, 'Time'] = df.loc[df['Id'] == citizen_id, 'Time'].shift(periods=start_ts)
        df = df.drop(df[(df['Started'] == 0)].index)
        df = df.drop(df[(df['Event'] == 1)].index)
        df = df.drop(aux_cols, axis=1)
        df = df.sort_values(by=["Id", "Time"])
        return df