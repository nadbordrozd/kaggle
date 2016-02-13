from train_utils import *
import traceback
    
info("start train.py $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

f_extractors = ['basic',
 'feats2',
 'fe2_km10',
 'fe2_km20',
 'fe2_km40',
 'oh_med_cut',
 'ohmed',         
 'oh_km10',
 'oh_km20',
 'oh_km40',
 'fe2_median'
 ]
     
f_extractors = ['basic', 'oh_med_cut', "fe2_km20", "fe2_median"]


models = [
    lin_lin_xgb,
    etr_lin_xgb,
    lin_mini_team,
    etr_mini_team,
    xgbr_mini_team,
    lin_mini_bayes,
    lin_mini_lasso,
    lin_mini_perc,
    lin_mini_svrrbf,
    lin_dream,
    lin_dream
]

for f in  ['oh_med_cut', "feats2"]:
    for m in models:
        try:
            result = lazy_benchmark(m(), f)
            info("ONE-FOLD CV got  %.3f    %s   kappa score, feats = %s" % (result, m(), f))
        except:
            s = traceback.format_exc()
            info("fucking error while training %s  on features %s" % (m(), f))
            info(s)

info("one - folders done")
for f in  ['oh_med_cut', "feats2"]:
    for m in models:
        try:
            result = benchmark(m(), f)
            info("FULL CV got  %.3f    %s   kappa score, feats = %s" % (result, m(), f))
        except:
            s = traceback.format_exc()
            info("fucking error while training %s  on features %s" % (m(), f))
            info(s)
        
info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")