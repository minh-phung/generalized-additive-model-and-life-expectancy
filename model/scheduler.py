import numpy as np
import pandas as pd

def main(predictor, target,
         corr,
         mi,
         folder, head = '', tail = ''):

    mi_nan = mi[mi.isna()].index.to_frame(index=False).to_numpy()[:,1]

    mi.dropna(inplace=True)
    mi_sorted = mi.iloc[np.argsort(mi)].index.to_frame(index=False).to_numpy()[:,1]

    abs_corr_sorted = corr.iloc[np.argsort(np.abs(corr))].index.to_frame(index=False).to_numpy()[:,1]

    half_predictor = len(predictor)//2
    variability = half_predictor//2

    all_schedule = []

    for i in range(half_predictor-variability, half_predictor+variability):
        each_schedule = np.full(len(predictor), 1.0)

        # ------------------
        for each_2 in abs_corr_sorted[-i:]:
            try:
                loc_2 = np.where(predictor == each_2)[0]
                each_schedule[loc_2] = 2
            except:
                raise ValueError("scheduler - abs corr comparison")
        # ------------------
        for j in range(0, half_predictor):
            for each_remove in mi_sorted[:j]:
                try:
                    loc_remove = np.where(predictor == each_remove)[0]
                    each_schedule[loc_remove] = -1
                except:
                    raise ValueError("scheduler - mi comparison")

            all_schedule.append(each_schedule.copy())


    schedule = pd.DataFrame(all_schedule, columns=predictor)

    schedule_unique = schedule.drop_duplicates()
    schedule_unique[target] = 't'


    schedule_unique.to_csv(folder + head + '_schedule_' + tail +".csv", index=False)

    return