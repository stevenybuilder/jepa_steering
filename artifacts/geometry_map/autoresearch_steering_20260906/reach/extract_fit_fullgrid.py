from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def pool(x,d):return x.float().reshape(-1,d).mean(0)
def features(context,pred,actions,h):
    a=actions[:h].float().reshape(-1,4)
    return torch.cat([pool(context['visual'],384),pool(context['proprio'],16),pool(pred['visual'],384),pool(pred['proprio'],16),a.mean(0),a[-1],a.sum(0)/15,torch.tensor([min(h,3)/3])])
def extract(args):
    torch.set_num_threads(2); rows=[];sources=[];began=time.time()
    for directory in args.sources:
        root=Path(directory);done=json.loads((root/'DONE.json').read_text());assert done['complete']
        entries={e['path']:e for e in done['outputs']}
        for p in sorted(root.glob('episode-*-unsteered-replan-*.pt')):
            e=int(p.name.split('-')[1]);assert 0<=e<12
            assert p.name in entries and sha(p)==entries[p.name]['sha256']
            t=torch.load(p,map_location='cpu',weights_only=False);assert t['episode']==e and t['arm']=='unsteered'
            for truth in t['actual_executed_prefix_encodings']:
                h=truth['horizon'];assert 1<=h<=3
                pred={k:t['selected_prediction'][k][h]for k in ('visual','proprio')}
                y=torch.cat([(truth['visual']-pred['visual']).reshape(256,384),(truth['proprio']-pred['proprio']).reshape(256,16)],1).flatten()
                rows.append((e,t['replan'],h,features(t['encoded_context'],pred,t['selected_full_plan'],h).numpy(),y.numpy()))
            sources.append({'path':str(p),'sha256':entries[p.name]['sha256'],'episode':e})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'rows.npz',X=np.stack([r[3]for r in rows]),Y=np.stack([r[4]for r in rows]),units=np.array([r[:3]for r in rows]))
    (out/'DONE.json').write_text(json.dumps(dict(complete=True,rows=len(rows),episodes=sorted(set(r[0]for r in rows)),sources=sources,output_sha256=sha(out/'rows.npz'),seconds=time.time()-began),indent=2))
    print(json.dumps(dict(complete=True,rows=len(rows),episodes=sorted(set(r[0]for r in rows)),seconds=time.time()-began)),flush=True)
def fit(args):
    torch.set_num_threads(2);d=[np.load(p)for p in args.sources];X=np.concatenate([x['X']for x in d]);Y=np.concatenate([x['Y']for x in d]);units=np.concatenate([x['units']for x in d]);assert len(set(map(tuple,units)))==len(units)
    mean=X.mean(0);scale=X.std(0).clip(.05);Z=(X-mean)/scale;Z=np.c_[np.ones(len(Z)),Z].astype(np.float64)
    # Strong fixed shrinkage; mean residual also penalized, no hyperparameter selection.
    ridge=10*len(Z);alpha=np.linalg.solve(Z@Z.T+ridge*np.eye(len(Z)),Y).astype(np.float32)
    gen=np.random.default_rng(20260906);perm=np.r_[gen.permutation(384),384+gen.permutation(16)];sign=gen.choice([-1.,1.],400)
    pred=(Z@Z.T)@alpha
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);np.savez(out/'correction.npz',mean=mean,scale=scale,train=Z.astype(np.float32),alpha=alpha,tokenperm=gen.permutation(256),perm=perm,sign=sign.astype(np.float32))
    (out/'DONE.json').write_text(json.dumps(dict(complete=True,episodes=sorted(set(units[:,0].tolist())),rows=len(X),ridge=ridge,inputs=[{'path':p,'sha256':sha(p)}for p in args.sources],train_native_fullgrid_mse=float(np.mean(Y**2)),train_corrected_fullgrid_mse=float(np.mean((Y-.5*pred)**2)),parameters=sha(out/'correction.npz')),indent=2));print((out/'DONE.json').read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();globals()[a.mode](a)
