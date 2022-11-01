import data
from hyperparams import *
from model import Transformer
from pathlib import Path
import train
from fns import two_digit_non_modular_mul

model = Transformer(
    num_layers=num_layers,
    d_vocab=d_vocab,
    d_model=d_model,
    d_mlp=d_mlp,
    d_head=d_head,
    num_heads=num_heads,
    n_ctx=4,
    act_type=act_type,
    use_cache=False,
    use_ln=use_ln,
)
# model.to("cuda")

if __name__ == "__main__":
    train.run_training(
        Path("."),
        "mul",
        two_digit_non_modular_mul,
        data.train,
        data.test,
        model,
        num_epochs=1,
    )
