import numpy as np

def L2Norm0(x1):
    return (x1.vertices**2).sum()

def L2NormDef(xDef, x1):
    #print(f'L2 norm: {np.abs(xDef.vertices - x1.vertices).max():.6f}')
    return -2*(xDef.vertices*x1.vertices).sum() + (xDef.vertices**2).sum()

def L2NormGradient(xDef,x1):
    return 2*(xDef.vertices-x1.vertices)


# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    cr2 = fv1.face_weights[:, None]
    return (cr2*KparDist.applyK(fv1.vertices, cr2)).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormDef(fvDef, fv1, KparDist):
    cr1 = fvDef.face_weights[:, None]
    cr2 = fv1.face_weights[:, None]
    obj = ((cr1 * KparDist.applyK(fvDef.vertices, cr1)).sum()
           - 2 * (cr1 * KparDist.applyK(fv1.vertices, cr2, firstVar=fvDef.vertices)).sum())
    return obj


# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    cr1 = fvDef.face_weights[:, None]
    cr2 = fv1.face_weights[:, None]

    dz1 = (KparDist.applyDiffKT(fvDef.vertices, cr1, cr1) -
                       KparDist.applyDiffKT(fv1.vertices, cr1, cr2, firstVar=fvDef.vertices))

    return 2 * dz1


