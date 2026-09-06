"""Locally anchored learned-chart interventions, not a certified native manifold.

PCA4 supplies coordinates. A fitted RBF decoder supplies nonlinear geometry.
Both controls begin at the exact recipient, use the same coordinate trust box,
and leave the enclosing PCA64 complement unchanged when lifted by the runner.
"""
import torch
from torch import nn


def squared_distances(x,y):
    return (x.square().sum(-1,keepdim=True)+y.square().sum(-1)[None]-2*x@y.T).clamp_min(0)


class LearnedChart(nn.Module):
    def __init__(self,mean,basis,coordinates,scale,epsilon,weights,lower,upper):
        super().__init__()
        for name,value in dict(mean=mean,basis=basis,coordinates=coordinates,scale=scale,
                               epsilon=epsilon,weights=weights,lower=lower,upper=upper).items():
            self.register_buffer(name,value.detach().double())

    @classmethod
    @torch.no_grad()
    def fit(cls,values):
        values=torch.as_tensor(values,dtype=torch.float64)
        if values.ndim!=2 or values.shape[1]!=64 or len(values)<10 or not torch.isfinite(values).all():
            raise ValueError('Expected finite donor rows in one common PCA64 basis')
        mean=values.mean(0);_,singular,vt=torch.linalg.svd(values-mean,full_matrices=False)
        if singular[3]<max(float(singular[0])*1e-8,1e-10):
            raise ValueError('Donor data do not support four coordinates')
        basis=vt[:4];c=(values-mean)@basis.T
        scale=c.std(0,unbiased=False).clamp_min(1e-8)
        distances=squared_distances(c,c);positive=distances[distances>1e-12]
        epsilon=torch.quantile(positive,.5)
        kernel=torch.exp(-distances/(2*epsilon))
        weights=torch.linalg.solve(kernel+1e-3*torch.eye(len(c),device=c.device,dtype=c.dtype),values-mean)
        return cls(mean,basis,c,scale,epsilon,weights,c.min(0).values,c.max(0).values)

    def encode(self,x):
        return (x.double()-self.mean)@self.basis.T

    def decode(self,c,nonlinear=True):
        c=c.double()
        if not nonlinear:return self.mean+c@self.basis
        return self.mean+torch.exp(-squared_distances(c,self.coordinates)/(2*self.epsilon))@self.weights

    def delta(self,x,control,nonlinear=True):
        c=self.encode(x)
        changed=c+.5*self.scale*torch.tanh(control.double())
        return self.decode(changed,nonlinear)-self.decode(c,nonlinear)

    def decode_graph(self,c):
        """Keep PCA4 coordinates exactly (up to double precision roundoff).

        Only the orthogonal part of the fitted RBF decoder supplies curvature.
        This is a graph over the chosen PCA4 plane, not a support certificate.
        """
        c=c.double()
        nonlinear=self.decode(c,nonlinear=True)-self.mean
        orthogonal=nonlinear-(nonlinear@self.basis.T)@self.basis
        return self.mean+c@self.basis+orthogonal

    def delta_graph(self,x,control):
        c=self.encode(x)
        changed=c+.5*self.scale*torch.tanh(control.double())
        return self.decode_graph(changed)-self.decode_graph(c)

    def support_coordinates(self,c):
        standardized=(c.double()-self.coordinates.mean(0))/self.scale
        training=(self.coordinates-self.coordinates.mean(0))/self.scale
        return dict(outside_train_box=((c<self.lower)|(c>self.upper)).any(-1),
                    nearest_train_distance=squared_distances(standardized,training).min(-1).values.sqrt())

    def support(self,x):
        return self.support_coordinates(self.encode(x))
