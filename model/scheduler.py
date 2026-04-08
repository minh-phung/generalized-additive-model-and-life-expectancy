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
        each_schedule = np.full(len(predictor), "l")

        # ------------------
        for each_2 in abs_corr_sorted[-i:]:
            try:
                loc_2 = np.where(predictor == each_2)[0]
                each_schedule[loc_2] = "s"
            except:
                raise ValueError("scheduler - abs corr comparison")
        # ------------------
        for j in range(0, half_predictor):
            for each_remove in mi_sorted[:j]:
                try:
                    loc_remove = np.where(predictor == each_remove)[0]
                    each_schedule[loc_remove] = "r"
                except:
                    raise ValueError("scheduler - mi comparison")
            for each_nan in mi_nan:
                try:
                    loc_nan = np.where(predictor == each_nan)[0]

                    each_schedule[loc_nan] = "l"
                    all_schedule.append(each_schedule.copy())

                    each_schedule[loc_nan] = "r"
                    all_schedule.append(each_schedule.copy())
                except:
                    raise ValueError("scheduler - mi nan comparison")


    schedule = pd.DataFrame(all_schedule, columns=predictor)

    schedule_unique = schedule.drop_duplicates().reset_index(drop=True)
    schedule_unique[target] = 't'
    schedule_unique.index.name = 'id'

    schedule_unique.to_csv(folder + head + tail +".csv", index=True)

    return