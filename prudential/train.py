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

benchmark_optimized_stacker = bos

bos(LinearRegression(), dream_team)

for i in range(len(dream_team())):
    dt = dream_team()
    new_team = dt[0:i] + dt[i+1:len(dt)]
    bos(LinearRegression(), new_team)
    


info("train.py done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::")