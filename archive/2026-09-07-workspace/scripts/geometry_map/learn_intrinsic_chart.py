#!/usr/bin/env python3
"""Exploratory unlabelled local charts; not a causal control or native-coordinate claim.

Density-normalized diffusion maps follow arXiv:math/0503445 section 4.
Fit each state's chart separately; do not mix incompatible state-specific PCA bases.
Entire action directions are withheld, but episodes are NOT held out across fits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def distances(x, y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)


def bandwidth(d):
    candidates=d[d>1e-12]
    if not len(candidates):raise ValueError('Degenerate identical inputs')
    return float(np.median(candidates))


class DiffusionChart:
    def __init__(self, rank, alpha):
        self.rank=rank;self.alpha=alpha

    def fit(self, x):
        self.x=np.asarray(x,dtype=np.float64)
        d=distances(self.x,self.x); self.epsilon=bandwidth(d)
        kernel=np.exp(-d/(2*self.epsilon));self.q=kernel.sum(1)
        affinity=kernel/(self.q[:,None]*self.q[None,:])**self.alpha
        degree=affinity.sum(1)
        symmetric=affinity/np.sqrt(degree[:,None]*degree[None,:])
        eigen,u=np.linalg.eigh(symmetric);order=np.argsort(eigen)[::-1]
        eigen=eigen[order];u=u[:,order]
        if self.rank>=len(eigen) or eigen[self.rank]<1e-8:
            raise ValueError('Requested diffusion rank unsupported')
        self.eigen=eigen[1:self.rank+1]
        self.psi=u[:,1:self.rank+1]/np.sqrt(degree[:,None])*np.sqrt(degree.sum())
        self.eigenvalues=eigen
        self.training=self.psi*self.eigen
        return self

    def transform(self,x):
        kernel=np.exp(-distances(x,self.x)/(2*self.epsilon))
        q=kernel.sum(1)
        affinity=kernel/(q[:,None]*self.q[None,:])**self.alpha
        p=affinity/affinity.sum(1,keepdims=True)
        # Nystrom psi_new=P_new psi/lambda, t=1 diffusion coords=lambda psi_new.
        return p@self.psi


def rbf_reconstruct(train_coordinates,test_coordinates,train_values):
    epsilon=bandwidth(distances(train_coordinates,train_coordinates))
    kernel=np.exp(-distances(train_coordinates,train_coordinates)/(2*epsilon))
    mean=train_values.mean(0)
    weights=np.linalg.solve(kernel+1e-3*np.eye(len(kernel)),train_values-mean)
    return np.exp(-distances(test_coordinates,train_coordinates)/(2*epsilon))@weights+mean


def evaluate(knots,linear_controls=False):
    knots=np.asarray(knots,dtype=np.float64)
    if knots.shape!=(4,4,9,64) or not np.isfinite(knots).all():
        raise ValueError('Expected four amplitudes x four directions x nine plans x PCA64')
    rows=[]
    for held_direction in range(4):
        train=knots[:,[d for d in range(4) if d!=held_direction]].reshape(-1,64)
        test=knots[:,held_direction].reshape(-1,64)
        mean=train.mean(0);_,_,vt=np.linalg.svd(train-mean,full_matrices=False)
        floor=float(np.mean((test-mean)**2))
        for rank in (2,4,8):
            pca_train=(train-mean)@vt[:rank].T;pca_test=(test-mean)@vt[:rank].T
            variants=[('pca',pca_train,pca_test,None)]
            for alpha in (0.,1.):
                chart=DiffusionChart(rank,alpha).fit(train)
                np.testing.assert_allclose(chart.transform(train),chart.training,atol=1e-9,rtol=1e-7)
                variants.append((f'diffusion_alpha_{alpha:g}',chart.training,chart.transform(test),chart))
            for method,tr,te,chart in variants:
                predictions=[('rbf',rbf_reconstruct(tr,te,train))]
                if linear_controls:
                    center=tr.mean(0)
                    weights=np.linalg.lstsq(tr-center,train-mean,rcond=1e-10)[0]
                    predictions.append(('linear',mean+(te-center)@weights))
                for decoder,pred in predictions:
                    mse=float(np.mean((pred-test)**2))
                    rows.append(dict(held_direction=held_direction,rank=rank,method=method,decoder_kind=decoder,
                    pca64_reconstruction_mse=mse,mean_baseline_mse=floor,
                    reduction_vs_mean=1-mse/floor if floor else None,
                    diffusion_eigenvalues=None if chart is None else chart.eigenvalues[:10].tolist(),
                    training_samples=len(train),testing_samples=len(test),independent_states=1,
                    split='within-state action-direction holdout, NOT episode-generalization',
                    decoder='fixed RBF ridge .001 or OLS linear control; same within each comparison; no outcome labels',
                    density_scope='Graph density of sampled activations, NOT model probability or belief'))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--linear-controls',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads(a.input.with_suffix('.json').read_text())
    source_sha=hashlib.sha256(a.input.read_bytes()).hexdigest()
    if not receipt.get('complete') or receipt['sha256']!=source_sha:
        raise ValueError('Input receipt mismatch before fit')
    protocol=dict(source_sha256=source_sha,source_receipt=receipt,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypotheses='Unlabelled local diffusion coordinates improve withheld-direction reconstruction versus equal-rank PCA',
        ranks=[2,4,8],diffusion_alpha=[0,1],diffusion_time=1,bandwidth='median positive TRAIN squared distance',
        splits='four leave-one-action-direction-out fits within this state',
        decoder='same RBF ridge .001, TRAIN-only bandwidth',linear_controls=a.linear_controls,
        endpoint='PCA64 reconstruction MSE, all methods/ranks/folds retained; no promotion to steering',
        compute='local one-thread CPU; no model/simulator/held episodes')
    a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=np.load(a.input,allow_pickle=False)
    rows=evaluate(data['knot_coordinates'],linear_controls=a.linear_controls)
    a.output.write_text(json.dumps(dict(complete=True,rows=rows,
        source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        source=str(a.input),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Local chart reconstruction only; not manifold identification, causal steering, or independent episode replication.'),indent=2,allow_nan=False)+'\n')
