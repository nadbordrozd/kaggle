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
     
for f in f_extractors:
    benchmark_model_optimized(xgbr(), f)

import traceback
for f in f_extractors:
    try:
        benchmark_model_optimized(linreg(), f)
    except:
        trace = traceback.format_exc()
        info("linreg FAILED  features = %s" % f)
        info(trace)
        
    
info("middle ,,,,,,,.....................,,,,,,,")
for f in f_extractors:
    info("training stacker with lin([lin, xgb])  feats = %s" % f)
    team = sorted([
            linreg(),
            xgbr()
           ])
    try:
        benchmark_optimized_stacker(linreg(), team, f)
    except:
        trace = traceback.format_exc()
        info("dream team FAILED  features = %s" % f)
        info(trace)



info("upper middle ,,,,,,,.....................,,,,,,,")
for f in f_extractors:
    info("training stacker with lin([lin, xgb, etr, svr])  feats = %s" % f)
    team = sorted([
            linreg(),
            xgbr(), 
            etr(),
            svrsig()
           ])
    try:
        benchmark_optimized_stacker(linreg(), team, f)
    except:
        trace = traceback.format_exc()
        info("dream team FAILED  features = %s" % f)
        info(trace)

info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")