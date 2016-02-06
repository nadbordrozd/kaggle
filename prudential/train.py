from train_utils import *

    
info("start train.py $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


#benchmark_model_optimized(xgbr())
#benchmark_model_optimized(etr())
#benchmark_model_optimized(LinearRegression())
#benchmark_optimized_stacker(xgbr(), dream_team())
#benchmark_optimized_stacker(etr(), dream_team())
#benchmark_optimized_stacker(rfr(), dream_team())
#benchmark_optimized_stacker(LinearRegression(), dream_team())
#benchmark_optimized_lazy_stacker(LinearRegression(), dream_team())
#benchmark_optimized_lazy_stacker(etr(), dream_team())
#benchmark_optimized_lazy_stacker(xgbr(), dream_team())

#info("making submission for xgbr optimized")
#make_sub_optimized(xgbr(), dream_team(), "xgbr_opt.csv")

#info("making submission for etr optimized")
#make_sub_optimized(etr(), dream_team(), "etr_opt.csv")

#info("making submission for linreg optimized")
#make_sub_optimized(LinearRegression(), dream_team(), "lin_opt.csv")

f_extractors = ['basic',
 'feats2',
 'fe2_km10',
 'fe2_km20',
 'fe2_km40',
 'oh_med_cut'
 'ohmed',         
 'oh_km10',
 'oh_km20',
 'oh_km40',

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


#bos = benchmark_optimized_stacker

#bos(LinearRegression(), dream_team())

#for i in range(len(dream_team())):
#    dt = dream_team()
#    info("testing dream team minus %s" % dt[i])
#    new_team = dt[0:i] + dt[i+1:len(dt)]
#    bos(LinearRegression(), new_team)
    


info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")