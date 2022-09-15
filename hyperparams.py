import numpy as np

lr = 1e-3
weight_decay = 1.0
p = 113
d_model = 128
# fn_name = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
frac_train = 0.3
num_epochs = 50000
save_models = True
save_every = 100
# Stop training when test loss is <stopping_thresh
stopping_thresh = -1
seed = 0

num_layers = 1
batch_style = "full"
d_vocab = p + 1
n_ctx = 3
d_mlp = 4 * d_model
num_heads = 4
assert d_model % num_heads == 0
d_head = d_model // num_heads
act_type = "ReLU"  # @param ['ReLU', 'GeLU']
use_ln = False
random_answers = np.random.randint(low=0, high=p, size=(p, p))
