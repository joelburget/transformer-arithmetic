import data
from hyperparams import *
from model import Transformer
from pathlib import Path
import train

model = Transformer(
    num_layers=num_layers,
    d_vocab=d_vocab,
    d_model=d_model,
    d_mlp=d_mlp,
    d_head=d_head,
    num_heads=num_heads,
    n_ctx=n_ctx,
    act_type=act_type,
    use_cache=False,
    use_ln=use_ln,
)
model.to("cuda")

if __name__ == "__main__":
    train.run_training(Path("."), "div", data.div_train, data.div_test, model)
