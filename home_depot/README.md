# kaggle_homedepot
Typical usage - benchmark two different models with two sets of features and make a submission.
```from benchmarks import benme, make_submission
import models as mo
import feature_engineering as fe

#modelname and featurename are optional
benme(mo.stacker_6(), fe.features_11, modelname="stacker 6", funname="bestest features ever")
benme(mo.stacker_6(), fe.features_12)
benme(mo.stacker_7(), fe.features_11)
benme(mo.stacker_7(), fe.features_12)

make_submission(mo.stacker_7(), fe.features_12, "submissions/stacker_7_feats_12.csv")
```

Save this as `run.py` and run with:
```screen -d -m -S benchmarks_and_shit /bin/bash -c "python run.py --data_dir fixed_data/ --cache_dir cache_fixed_data/ --log_path log_fixed_data.log 2> errorfile.txt"```

This:
- runs the script in a separate process (so it doesn't die when your connection to aws dies if you're running on aws)
- redirects error output to errorfile.txt (otherwise it would be lost since it's in a separate process)
- sets script parameters specifying where to look for input data, disk cache and log file