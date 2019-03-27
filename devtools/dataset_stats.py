import pandas as pd
import numpy as np

LABELS_AS_TXT = [
    'Non-Bullying',
    'Cultural',
    'Sexual',
    'Personal'
]

data_frame = pd.read_csv('../Data/new_data.csv')
labels, label_counts = np.unique(np.array(data_frame['label']), return_counts=True)
labels = list(map(lambda lab_num: LABELS_AS_TXT[lab_num], labels))
label_pcts = (label_counts / data_frame.shape[0]) * 100.0
label_pcts = list(map(lambda pct_val: '{:.01f}'.format(pct_val), label_pcts))
results = pd.DataFrame(
    {
        'Labels': labels,
        'Count': label_counts,
        'Percentage': label_pcts
    }
)

print(results)
