import numpy as np
import einops
import torch
import torch.nn.functional as F
import pandas as pd
import data
from hyperparams import p


# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated() / 1e9)


def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def full_loss(fn, model, data):
    # Take the final position only
    logits = model(data)[:, -1]
    labels = torch.tensor([fn(i, j) for i, j, _ in data]).to("cuda")
    return cross_entropy_high_precision(logits, labels)


def test_logits(
    logits,
    bias_correction=False,
    original_logits=None,
    mode="all",
    is_train=data.is_train,
    is_test=data.is_test,
):
    # Calculates cross entropy loss of logits representing a batch of all p^2
    # possible inputs
    # Batch dimension is assumed to be first
    if logits.shape[1] == p * p:
        logits = logits.T
    if logits.shape == torch.Size([p * p, p + 1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p * p, p)
    if bias_correction:
        # Applies bias correction - we correct for any missing bias terms,
        # independent of the input, by centering the new logits along the batch
        # dimension, and then adding the average original logits across all inputs
        logits = (
            einops.reduce(original_logits - logits, "batch ... -> ...", "mean") + logits
        )
    if mode == "train":
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode == "test":
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode == "all":
        return cross_entropy_high_precision(logits, labels)


def unflatten_first(tensor):
    if tensor.shape[0] == p * p:
        return einops.rearrange(tensor, "(x y) ... -> x y ...", x=p, y=p)
    else:
        return tensor


def cos(x, y):
    return (x.dot(y)) / x.norm() / y.norm()


def mod_div(a, b):
    return (a * pow(b, p - 2, p)) % p


def normalize(tensor, axis=0):
    return tensor / (tensor).pow(2).sum(keepdim=True, axis=axis).sqrt()


def extract_freq_2d(tensor, freq):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2 * freq - 1, 2 * freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    return tensor[[[i] * 3 for i in index_1d], [index_1d] * 3]


def get_cov(tensor, norm=True):
    # Calculate covariance matrix
    if norm:
        tensor = normalize(tensor, axis=1)
    return tensor @ tensor.T


def is_close(a, b):
    return (
        (a - b).pow(2).sum() / (a.pow(2).sum().sqrt()) / (b.pow(2).sum().sqrt())
    ).item()


neel_fourier_basis = []
neel_fourier_basis.append(torch.ones(p) / np.sqrt(p))
neel_fourier_basis_names = ["Const"]
# Note that if p is even, we need to explicitly add a term for cos(kpi), ie
# alternating +1 and -1
for i in range(1, p // 2 + 1):
    neel_fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(p) * i / p))
    neel_fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(p) * i / p))
    neel_fourier_basis[-2] /= neel_fourier_basis[-2].norm()
    neel_fourier_basis[-1] /= neel_fourier_basis[-1].norm()
    neel_fourier_basis_names.append(f"cos {i}")
    neel_fourier_basis_names.append(f"sin {i}")
neel_fourier_basis = torch.stack(neel_fourier_basis, dim=0).to("cuda")


sin_fourier_basis = []
sin_fourier_basis.append(torch.ones(p) / np.sqrt(p))
sin_fourier_basis_names = ["Const"]
for i in range(1, p):
    x = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
    sin_fourier_basis.append(x / x.norm())
    sin_fourier_basis_names.append(f"sin {i}")
sin_fourier_basis = torch.stack(sin_fourier_basis, dim=0).to("cuda")


def fft1d(fourier_basis, tensor):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ fourier_basis.T


def fourier_2d_basis_term(fourier_basis, x_index, y_index):
    # Returns the 2D Fourier basis term corresponding to the outer product of
    # the x_index th component in the x direction and y_index th component in the
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()


def fft2d(fourier_basis, mat):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, "(x y) ... -> x y (...)", x=p, y=p)
    fourier_mat = torch.einsum("xyz,fx,Fy->fFz", mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)


def analyse_fourier_2d(tensor, top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append(
            [
                tensor.flatten()[indices[i]].item(),
                values[i].item() / total,
                values[: i + 1].sum().item() / total,
                neel_fourier_basis_names[indices[i].item() // p],
                neel_fourier_basis_names[indices[i] % p],
            ]
        )
    display(
        pd.DataFrame(
            rows,
            columns=[
                "Coefficient",
                "Frac explained",
                "Cumulative frac explained",
                "x",
                "y",
            ],
        )
    )


def get_2d_fourier_component(tensor, x, y):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component
    # (x, y)
    vec = fourier_2d_basis_term(x, y).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)


def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2 * freq - 1, 2 * freq - 1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2 * freq, 2 * freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction) / np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return cos_xpy_direction @ tensor
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)


def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2 * freq, 2 * freq - 1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2 * freq - 1, 2 * freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction) / np.sqrt(2)
    if collapse_dim:
        return sin_xpy_direction @ tensor
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)
