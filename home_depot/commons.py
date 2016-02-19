#=========================================LOGGING
import logging
# create logger
logging.basicConfig(filename='.log',level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("logginho")
logger.setLevel(logging.DEBUG)

from joblib import Memory
cache = Memory(cachedir="cache_dir/").cache

def get_func_name(func):
    try:
        return func.func_name
    except AttributeError:
        return func.func.func_name
