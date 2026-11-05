"""Make equal-size, episode-disjoint train/validation dataset variants.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def choose(meta,
           n,
           rng):

    if len(meta) < 2*n:
        raise ValueError('Need at least twice --per-group episodes to form disjoint top/bottom sets.')

    ranked = meta.sort_values(['final_distance','episode_id'])
    high = ranked.head(n).episode_id.to_numpy()
    low = ranked.tail(n).episode_id.to_numpy()
    random_ids = rng.choice(meta.episode_id.to_numpy(), size=n, replace=False)

    return {'high_quality':high,
            'low_quality':low,
            'random_baseline':random_ids}


def build(source,
          out,
          n,
          val_fraction,
          seed):

    rng = np.random.default_rng(seed)
    source = Path(source); out = Path(out)
    meta = pd.read_csv(source/'episode_metadata.csv')
    with np.load(source/'trajectories.npz') as d:
        obs, acts, episode_ids = d['observations'], d['actions'], d['episode_ids']

    variants = choose(meta, n, rng)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for variant, ids in variants.items():
        ids = rng.permutation(ids)
        val_n = max(1, int(round(len(ids)*val_fraction)))
        if val_n >= len(ids):
            raise ValueError('Validation split leaves no training episodes.')
        splits = {'val':ids[:val_n], 'train':ids[val_n:]}
        for split, chosen in splits.items():
            mask = np.isin(episode_ids, chosen)

            np.savez_compressed(out/f'{variant}_{split}.npz',
                                observations=obs[mask],actions=acts[mask],
                                episode_ids=episode_ids[mask])

            records.append(dict(variant=variant,split=split,episodes=len(chosen),
                                transitions=int(mask.sum())))

    pd.DataFrame(records).to_csv(out/'manifest.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('data/expert'))
    p.add_argument('--out',type=Path,default=Path('data/variants'))
    p.add_argument('--per-group',type=int,default=80)
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=123)
    a=p.parse_args()
    if a.per_group < 2 or not 0<a.val_fraction<1:
        p.error('--per-group >= 2 and 0 < --val-fraction < 1 required')
    build(a.source,
          a.out,
          a.per_group,
          a.val_fraction,
          a.seed)

if __name__=='__main__':main()
