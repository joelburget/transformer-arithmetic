import sys
import time
import os

import einops
import torch
import torch.optim as optim

from model import Mlps, NoMlp, Transformer
import plotting
from hyperparams import *
import util


def run_training(
    root, fn_name, fn, train_data, test_data, model, num_epochs=num_epochs
):
    if model is None:
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

    # model.to("cuda")
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
    run_name = f"grok_{int(time.time())}"
    print(f"Run name {run_name}")
    if save_models:
        os.mkdir(root / run_name)
        save_dict = {
            "model": model.state_dict(),
            "train_data": train_data,
            "test_data": test_data,
        }
        torch.save(save_dict, root / run_name / f"{fn_name}-init.pth")
    train_losses = []
    test_losses = []
    epochs = []
    state_dicts = []
    for epoch in range(num_epochs):
        train_loss = util.full_loss(fn, model, train_data)
        test_loss = util.full_loss(fn, model, test_data)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        if epoch % 100 == 0:
            epochs.append(epoch)
            state_dicts.append(model.state_dict())
            print(
                f"\r{epoch}_{np.log(train_loss.item()):.4f}_{np.log(test_loss.item()):.4f}",
                end="",
            )
            # print(f"{epoch}_{np.log(train_loss.item()):.4f}_{np.log(test_loss.item()):.4f}")#_{train_acc.item():.4f}_{test_acc.item():.4f}")
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if test_loss.item() < stopping_thresh:
            break
        if (save_models) and (epoch % save_every == 0):
            if test_loss.item() < stopping_thresh:
                break
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch": epoch,
            }
            torch.save(save_dict, root / run_name / f"{fn_name}-{epoch}.pth")
            # print(f"Saved model to {root/run_name/f'{fn_name}-{epoch}.pth'}")
    if not save_models:
        os.mkdir(root / run_name)
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "epoch": epoch,
    }
    torch.save(save_dict, root / run_name / f"{fn_name}-final.pth")
    print(f"Saved final model to {root/run_name/f'{fn_name}-final.pth'}")
    torch.save(
        {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "epochs": epochs,
            "state_dicts": state_dicts,
            "model": model,
            # 'config': lr, p, etc
        },
        root / run_name / f"{fn_name}-full-run.pth",
    )
    plotting.lines([train_losses, test_losses], labels=["train", "test"], log_y=True)
