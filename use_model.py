import pickle

import numpy as np

local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

new_pred = local_classifier.predict(local_scaler.transform(np.array([[40,20000]])))

new_pred_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[40,20000]])))[:,1]


new_pred_2 = local_classifier.predict(local_scaler.transform(np.array([[42,50000]])))

new_pred_proba_2 = local_classifier.predict_proba(local_scaler.transform(np.array([[42,50000]])))[:,1]
