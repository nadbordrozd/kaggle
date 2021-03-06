from train_utils import *
import traceback
    
info("start train.py $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

m = lin_mini_team
for f in ["ohmedcut_kw_nan_poly015"]:
    make_sub(lin_mini_team(), f, f + ".csv")
    result = lazy_benchmark(m(), f)
    info("ONE-FOLD CV got  %.3f    %s   kappa score, feats = %s" % (result, m(), f))


0/0


f_extractors = [
    'ohmedcut_kw_nan',
    'ohmedcut_kw_nan_poly02',
    'ohmedcut_kw_nan_poly015',
    'ohmedcut_kw_nan_poly012',
    'ohmedcut_poly02',
    'ohmedcut_poly015',
    'ohmedcut_poly012',
    'basic',     
    'oh_med_cut',
    'ohmedcut_kmns10',
    'ohmedcut_kmns20',
    'ohmedcut_kmns40',
    'ohmed', 
    "feats2", 
    "fe2_median", 
    'fe2_kmns10',
    'fe2_kmns20',
    'fe2_kmns40',
    'ohmedcut_kwcount',
    'ohmedcut_nancount'
]


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

for m in models:
    for f in f_extractors:
        try:
            result = lazy_benchmark(m(), f)
            info("ONE-FOLD CV got  %.3f    %s   kappa score, feats = %s" % (result, m(), f))
        except:
            s = traceback.format_exc()
            info("fucking error while training %s  on features %s" % (m(), f))
            info(s)

info("one - folders done")
for f in f_extractors:
    for m in models:
        try:
            result = benchmark(m(), f)
            info("FULL CV got  %.3f    %s   kappa score, feats = %s" % (result, m(), f))
        except:
            s = traceback.format_exc()
            info("fucking error while training %s  on features %s" % (m(), f))
            info(s)
        
info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")