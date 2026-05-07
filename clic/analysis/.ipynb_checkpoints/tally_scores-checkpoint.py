import numpy as np
import pandas as pd

def IQR(column): 
    q25, q75 = column.quantile([0.25, 0.75])
    return q75-q25

def statistics(dataframe:pd.DataFrame,targets:list[str]):

    datastats = dataframe.groupby(targets)["accuracy"].agg(["mean","median","count","std",IQR])

    ci95_hi = []
    ci95_lo = []

    for i in datastats.index:
        m,_, c, s, _ = datastats.loc[i]
        ci95_hi.append(m + 1.645*s/np.sqrt(c))
        ci95_lo.append(m - 1.645*s/np.sqrt(c))

    datastats['ci95_hi'] = ci95_hi
    datastats['ci95_lo'] = ci95_lo

    datastats = datastats.reset_index()

    return datastats