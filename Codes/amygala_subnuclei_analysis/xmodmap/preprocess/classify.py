import sys
import os
import torch
from matplotlib import pyplot as plt
sys.path.append('..')
sys.path.append('../..')
import xmodmap

from xmodmap.io.getOutput import writeVTK

trainingFile = sys.argv[1]
savedir = sys.argv[2]
if len(sys.argv) > 3:
    suff = sys.argv[3]
else:
    suff = ""

bc = xmodmap.preprocess.BoundaryClassifier()
bc.train(trainingFile)
bc.save(os.path.join(savedir,suff + "boundaryModel.pt"))
#print(loss)
#f,ax = plt.subplots()
#ax.plot(loss)
#f.savefig(os.path.join(savedir,"boundaryModelLoss.png"),dpi=300)

T = torch.load(trainingFile)['X']

# plot grid 
mi = torch.min(T,axis=0).values
ma = torch.max(T,axis=0).values
xx = torch.arange(mi[0]-3.0,ma[0]+1.0,0.25)
yy = torch.arange(mi[1]-3.0,ma[1]+1.0,0.25)
zz = torch.arange(mi[2]-3.0,ma[2]+1.0,0.25)

XX,YY,ZZ = torch.meshgrid(xx,yy,zz,indexing='ij')
XXYYZZ = torch.stack((XX.ravel(),YY.ravel(),ZZ.ravel()),axis=-1)
swGrid1 = bc.predict(1.0,XXYYZZ)

writeVTK(XXYYZZ,[swGrid1.cpu().numpy()],['supportWeights1'],os.path.join(savedir,suff+"initialSupportWeights.vtk"))
