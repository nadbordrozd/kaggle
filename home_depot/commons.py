import sys
import argparse

parser = argparse.ArgumentParser(description='do stuff')
parser.add_argument("--data_dir", default="data/", help='directory to look for train.csv and test.csv')
parser.add_argument("--cache_dir", default="cache/", help="(mostly?) feature engineering cache dir")
parser.add_argument("--log_path", default="run_log.log", help="what it says on the tin")

parser.add_argument("-f")
args = parser.parse_args()
CACHE_DIR = args.cache_dir
DATA_DIR = args.data_dir
LOG_PATH = args.log_path

#=========================================LOGGING
import logging
# create logger
logging.basicConfig(filename=LOG_PATH,level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("logginho")
logger.setLevel(logging.DEBUG)

#MEMOIZATION
from joblib import Memory
logger.info("args")
logger.info(args)
cache = Memory(cachedir=CACHE_DIR).cache

def get_func_name(func):
    try:
        return func.func_name
    except AttributeError:
        return func.func.func_name
