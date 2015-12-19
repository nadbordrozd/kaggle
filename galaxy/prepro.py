import sys
import cPickle as pickle
from numpy import nanmean
import numpy as np
from scipy import misc, ndimage
import glob
import logging

FORMAT = '%(asctime)s %(levelname)s %(name)s:  %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('preprocessing')
logger.setLevel("INFO")


def to_bw(image):
    return np.mean(image, axis=-1)

def downsample(myarr,factor,estimator=nanmean):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr

def crop(image, dimx, dimy):
    old_dimx, old_dimy = image.shape
    fromx, tox = (old_dimx - dimx)/2, (old_dimx + dimx)/2
    fromy, toy = (old_dimy - dimy)/2, (old_dimy + dimy)/2
    return image[fromx:(tox + 1),fromy:(toy + 1)]

def to01(image):
    return image/256

def process_img(path):
    galaxy = misc.imread(path)
    return to01(crop(downsample(to_bw(galaxy), 3), 64, 64))

def process_chunk(img_paths, target_path):
    logger.info("processing chunk")
    images = np.array([process_img(path) for path in img_paths])
    X = images.astype(np.float32).reshape(-1, 1, 65, 65)
    logger.info("saving chunk to %s" % target_path)
    with open(target_path, "wb") as out:
        pickle.dump(X, out)
    logger.info("done with the chunk")

def prepare_chunks(paths, target_path, chunk_size):
    n = len(paths)
    groups_of_paths = [paths[i:i+chunk_size] for i in xrange(0, n, chunk_size)]
    for chunk_num, group in enumerate(groups_of_paths):
        chunk_path = "%s.%04.d.pickle" % (target_path, chunk_num)
        process_chunk(group, chunk_path)

        
if __name__ == "__main__":
    filenames = sorted(glob.glob(sys.argv[1]))
    indices = [int(f.split("/")[-1].strip(".jpg")) for f in filenames]
    target_path = sys.argv[2]
    chunk_size = int(sys.argv[3])
    logger.info("START")
    prepare_chunks(filenames, target_path, chunk_size)
    logger.info("done and done")
