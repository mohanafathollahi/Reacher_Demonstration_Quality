"""Train three-seed behavioral cloning baseline with validation early selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common import BCPolicy, set_seed


def load_data(path):
    with np.load(path) as d:
        return (torch.from_numpy(d['observations'].copy()).float(),
                torch.from_numpy(d['actions'].copy()).float())


def train(dataset,
          variant,
          seed,
          epochs,
          batch_size,
          lr,
          out):
    set_seed(seed)
    x,y = load_data(dataset/f'{variant}_train.npz')
    vx,vy = load_data(dataset/f'{variant}_val.npz')
    model=BCPolicy()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator().manual_seed(seed)

    best=float('inf'); history=[]
    out.mkdir(parents=True,exist_ok=True)
    save=out/f'{variant}_seed{seed}.pt'

    for epoch in range(1,epochs+1):
        model.train()
        perm=torch.randperm(len(x),generator=generator)

        for inds in perm.split(batch_size):
            loss=torch.nn.functional.mse_loss(model(x[inds]),y[inds])
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss=float(torch.nn.functional.mse_loss(model(vx),vy))
            training_loss=float(torch.nn.functional.mse_loss(model(x),y))
        history.append(dict(variant=variant,seed=seed,epoch=epoch,
                            train_mse=training_loss,val_mse=validation_loss))
        if validation_loss < best:
            best=validation_loss
            torch.save(model.state_dict(),save)
    pd.DataFrame(history).to_csv(out/f'{variant}_seed{seed}_history.csv',index=False)
    print(f'{variant} seed={seed}: best validation MSE={best:.6f} -> {save}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('data/variants'))
    p.add_argument('--out',type=Path,default=Path('models'))
    p.add_argument('--variants',nargs='+',default=['high_quality','low_quality','random_baseline'])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args()

    for v in a.variants:
        for seed in a.seeds:
            train(a.dataset,
                  v,
                  seed,
                  a.epochs,
                  a.batch_size,
                  a.lr,
                  a.out)

if __name__=='__main__':main()
