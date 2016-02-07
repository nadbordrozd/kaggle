from train_utils import *

    
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
     
f_extractors = ['basic', 'oh_med_cut', "fe2_km20", "fe2_km10"]
models = [st_lazy_hard, st_hard_lazy, st_hard_hard, st_lazy_lazy]
for f in f_extractors:
    for m in models:
        result = benchmark(m(), f)
        info("FULL CV %s  got  %.3f  kappa score, feats = %s" % (m(), result, f))
        result = lazy_benchmark(m(), f)
        info("ONE-FOLD CV %s  got  %.3f  kappa score, feats = %s" % (m(), result, f))

info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")