import numpy as np
from LibVQ.utils import evaluate
from LibVQ.utils import save_to_SPTAG_binary_file


from LibVQ.base_index import FaissIndex
# index = FaissIndex()
# index.load_index('/ads-nfs/t-shxiao/LibVQ/examples/SPTAG_Backbone/Results/ARG/opq_ivf-1_pq32x8.index')

# save_to_SPTAG_binary_file(index, 'input_for_SPTAG/OPQ')
import os, struct

rotate = np.load('/ads-nfs/t-shxiao/LibVQ/examples/SPTAG_Backbone/saved_ckpts/distill-VQ/epoch_29_step_5940/rotate_matrix.npy')
rotate_matrix = rotate.T

codebooks = np.load('/ads-nfs/t-shxiao/LibVQ/examples/SPTAG_Backbone/saved_ckpts/distill-VQ/epoch_29_step_5940/codebook.npy')

with open(os.path.join('input_for_SPTAG/LearnableIndex', 'index_parameters.bin'), 'wb') as f:
    f.write(struct.pack('B', 2))
    f.write(struct.pack('B', 3))
    f.write(struct.pack('i', codebooks.shape[0]))
    f.write(struct.pack('i', codebooks.shape[1]))
    f.write(struct.pack('i', codebooks.shape[2]))
    f.write(codebooks.tobytes())
    f.write(rotate_matrix.tobytes())
