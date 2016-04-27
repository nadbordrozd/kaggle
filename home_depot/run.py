import feature_engineering as fe
import benchmarks as ben
import models as mo
from commons import logger, get_func_name, DATA_DIR


logger.info("========================= start")


ben.benme(mo.stacker_7, fe.features_12, "stacker8", "features12")

logger.info("======================= end")