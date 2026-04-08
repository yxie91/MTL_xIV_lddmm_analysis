import os
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import scipy
import numpy as np
from numpy import pi
import h5py
import qpsolvers as qp
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt, cm
from scipy.optimize import nnls, lsq_linear, linprog


def tridiag_solve(a, b, c):
    n = b.shape[0]
    c2 = np.zeros(b.shape)
    d2 = np.zeros([b.shape[0], b.shape[0]] + list(b.shape[1:]))
    res = np.zeros(d2.shape)
    c2[0, ...] = c[0,...]/b[0,...]
    for j in range(1,n-1):
        c2[j,...] = c[j, ...] / (b[j, ...] - a[j-1, ...]*c2[j-1, ...])

    d = np.reshape(np.eye(b.shape[0]), [n, n] + [1] * (b.ndim-1))
    d2[0, ...] = d[0,...]/b[0,None, ...]
    for j in range(1,n):
        d2[j,...] = (d[j, ...] - a[j-1,None, ...] * d2[j-1, ...])\
                    / (b[j, None, ...] - a[j-1, None, ...]*c2[j-1, None, ...])

    res[n-1, ...] = d2[n-1, ...]
    for j in range(n-2, -1, -1):
        res[j, ...] = d2[j, ...] - c2[j, None, ...]*res[j+1, ...]

    # test = b[:, None, ...] * res
    # test[1:, :, ...] += a[:, None, ...] * res[:-1, :, ...]
    # test[:-1, :, ...] += c[:, None, ...] * res[1:,:,...]


    return res



dim = 2
sigma = 1
scales = np.linspace(.1, 2., 20)
#width = np.array([.05, .05])
width = np.sqrt(scales[:-1])
n = len(scales)-1
#weight = width**(2)
weight = np.ones(n)
freq = np.linspace(0, 10.0, 1001)
nf = freq.shape[0]
#ds = np.zeros(n-1)
# ds = np.zeros(n)
# chi = np.zeros((n+1, freq.shape[0]))
# psi = np.zeros((n+1, freq.shape[0]))
# for k in range(n):
#     ds[k] = scales[k+1] - scales[k]
# for k in range(n+1):
#     chi[k, :] = weight[k] *(2*pi*width[k]**2)**(dim/2) * np.exp(2*(pi*width[k]*freq)**2)
# for k in range(1, n):
#     psi[k, :] = (weight[k-1]/weight[k]) *(width[k-1]/width[k])**dim \
#                 * np.exp(2*(pi*width[k-1]*freq)**2 - 2*(pi*width[k]*freq)**2)
# psi[n, :] = 1
# a = np.zeros((n, nf))
# b = np.zeros((n+1, nf))
# c = np.zeros((n, nf))
# d0 = np.cosh(ds) / np.sinh(ds)
# b[0, :] = d0[0]
# for k in range(1, n):
#     b[k, :] = d0[k] + d0[k-1] * psi[k, :]
# b[n, :] = d0[n-1]
# d0 = 1 / np.sinh(ds)
# for k in range(n):
#     c[k, :] = d0[k] * psi[k+1, :]
#     a[k, :] = d0[k]


ds = sigma*(scales[1:] - scales[:-1])
d0 = np.cosh(ds) / np.sinh(ds)
invchi = (1/weight[:, None])*(2*pi*width[:, None]**2)**(dim/2) * np.exp(-2*(pi*width[:,None]*freq[None, :])**2)
psi = np.zeros((n+1, freq.shape[0]))
psi[1:-1, :] = (weight[:-1, None]/weight[1:, None])*(width[1:, None]/width[:-1, None])**dim \
           * np.exp(-2*(pi**2)*(width[1:,None]**2 - width[:-1,None]**2)*freq[None, :]**2)
psi[-1, :] = 1
h = np.zeros((n+1, n+1, freq.shape[0]))

# a = np.zeros((n, nf))
b = np.zeros((n+1, nf))
# c = np.zeros((n, nf))
b[:-1, :] = d0[:, None]
b[1:, :] += d0[:, None] * psi[1:, :]
c = -psi[1:, :] / np.sinh(ds[:, None])
a = - 1 / np.sinh(ds[:, None])
g = tridiag_solve(a, b, c)
g[:-1, :, :] *= invchi[:,None, :] / sigma
g[-1, :, :] *= invchi[-1, :] / sigma

plt.figure()
for l in range(0, g.shape[1]):
    plt.plot(freq, g[0, l, :], label=f'{l}')
plt.legend()
plt.title('FT at scale s1')
plt.figure()
for l in range(0, g.shape[1]):
    plt.plot(freq, g[-1, l, :], label=f'{l}')
plt.legend()
plt.title('FT at scale s2')
#plt.show()

t = np.sqrt(np.linspace(0.001, 10, 200))
nt = t.shape[0]
kappa_pos = np.zeros((n+1, n+1, nt))
kappa = np.zeros((n+1, n+1, nt))
A = np.exp(- (pi * freq[:, None]) ** 2 / t[None, :]) * (pi / t[None, :]) ** (dim / 2)
#Af = freq[:, None] * A
x = np.linspace(-5, 5, 201)
AA = np.exp(- x[:, None] ** 2 * t[None, :])
G = np.zeros((3 * A.shape[0], A.shape[1]+1))
G[:A.shape[0], :-1] = A
G[:A.shape[0], -1] = -1
G[A.shape[0]:2*A.shape[0], :-1] = -A
G[A.shape[0]:2*A.shape[0], -1] = -1
G[2*A.shape[0]:, :-1] = -A

h = np.zeros(g.shape)
hpos = np.zeros(g.shape)
eps0 = .01
K1 = np.zeros((n+1, n+1, x.shape[0]))
K2 = np.zeros((n+1, n+1, x.shape[0]))
for i1 in range(n+1):
    b = g[i1, i1, :]
    # eps = eps0
    c = np.zeros(G.shape[1])
    c[-1] = 1
    # while not success:
    b_ub = np.zeros(3*b.shape[0])
    b_ub[:b.shape[0]] = np.maximum(b, 1e-10)
    b_ub[b.shape[0]:2*b.shape[0]] = -b
    kappa_pos[i1, i1, :], err_pos = nnls(A, b)
    optres = linprog(c, A_ub = G, b_ub = b_ub)
    success = optres.success
#        eps = 1.1*eps
    kappa[i1, i1, :] = optres.x[:-1]
    eps = optres.x[-1]
    # kappa[i1, i1, :] = qp.solve_ls(A, b, G=-A, h=np.zeros(b.shape), solver = 'clarabel')
    err = np.abs(A@kappa[i1, i1, :] - b).max()

    if True or err > 0.01:
        print(f'{i1}, {i1}: eps = {eps:.04f}; error = {err:.4f}; error with positivity constraint {err_pos: .4f}')
    print(i1, i1, (np.abs(kappa[i1, i1, :])>1e-6).sum())
    h[i1, i1, :] = A@kappa[i1, i1, :]
    hpos[i1, i1, :] = A@kappa_pos[i1, i1, :]
    K1[i1, i1, :] = AA @ kappa[i1, i1, :]
    K2[i1, i1, :] = AA @ kappa_pos[i1, i1, :]

G = np.zeros((4 * A.shape[0], A.shape[1]+1))
G[:A.shape[0], :-1] = A
G[:A.shape[0], -1] = -1
G[A.shape[0]:2*A.shape[0], :-1] = A
G[2*A.shape[0]:3*A.shape[0], :-1] = -A
G[2*A.shape[0]:3*A.shape[0], -1] = -1
G[3*A.shape[0]:, :-1] = -A
c = np.zeros(G.shape[1])
c[-1] = 1
for i1 in range(n+1):
    for i2 in range(i1+1, n+1):
        b = g[i1, i2, :]
        bds = np.sqrt(np.maximum(h[i1, i1, :]*h[i2, i2, :], 1e-10))
        eps = eps0
        q = np.zeros(4*A.shape[0])
        # success = False
        # while not success:
        q[:A.shape[0]] = b
        q[A.shape[0]:2*A.shape[0]] = bds
        q[2*A.shape[0]:3*A.shape[0]] = -b
        q[3*A.shape[0]:] = bds
        bds_pos = np.sqrt(np.maximum(kappa_pos[i1, i1, :]*kappa_pos[i2, i2, :], 1e-10))
        # opt = lsq_linear(A, b, bounds=(-bds_pos, bds_pos))
        optres = linprog(c, A_ub=G, b_ub=q)
        # success = optres.success
        eps = 1.1 * eps
        # kappa_pos[i1, i2, :] = optres.x   #nnls(A, b)
        # err_pos = opt.cost
        kappa[i1, i2, :] = optres.x[:-1]
        eps = optres.x[-1]
        #kappa[i1, i2, :] = qp.solve_ls(A, b, G=G, h=q, solver = 'clarabel')
        err = np.abs(A@kappa[i1, i2, :] - b).max()
        kappa_pos[i1, i2, :] = kappa[i1, i2, :]
        err_pos = err

        if True or err > 0.01:
            print(f'{i1}, {i2}: eps = {eps:.04f}; error = {err:.4f}; error with positivity constraint {err_pos: .4f}')
        h[i1, i2, :] = A@kappa[i1, i2, :]
        hpos[i1, i2, :] = A@kappa_pos[i1, i2, :]
        K1[i1, i2, :] = AA @ kappa[i1, i2, :]
        K2[i1, i2, :] = AA @ kappa_pos[i1, i2, :]
for i1 in range(n+1):
    for i2 in range(i1):
        kappa[i1, i2, :] = kappa[i2, i1, :]
        K1[i1, i2, :] = K1[i2, i1, :]
        K2[i1, i2, :] = K2[i2, i1, :]
        kappa_pos[i1, i2, :] = kappa_pos[i2, i1, :]
        h[i1, i2, :] = h[i2, i1, :]
        hpos[i1, i2, :] = hpos[i2, i1, :]


plt.figure()
for l in range(0, g.shape[1]):
    plt.plot(freq, h[0, l, :], label=f'{l}')
plt.legend()
plt.title('Approx FT at scale s1')

plt.figure()
for l in range(0, g.shape[1]):
    plt.plot(freq, h[-1, l, :], label=f'{l}')
plt.legend()
plt.title('Approx FT at scale s2')

plt.figure()
plt.plot(freq, g[10, 10, :], label=f'g')
plt.plot(freq, h[10, 10, :], label=f'h')
plt.plot(freq, hpos[10, 10, :], label=f'hpos')
plt.legend()
plt.title('Comparison FT at scale 10')

plt.figure()
plt.plot(x, K1[10, 10, :], label=f'K1')
plt.plot(x, K2[10, 10, :], label=f'K2')
plt.legend()
plt.title('Kernels at scale 10')

s = np.linspace(scales[0], scales[-1], 201)
js = 0
bs = 10
img1 = np.zeros((s.shape[0], x.shape[0]))
img2 = np.zeros((s.shape[0], x.shape[0]))
for j in range(s.shape[0]):
    if s[j] > scales[js+1]:
        js += 1
    img1[j, :] = (np.sinh(sigma * (scales[js+1] - s[j])) * K1[js, bs, :] + np.sinh(sigma * (s[j] - scales[js])) * K1[js+1, bs, :]) \
                / np.sinh(sigma * (scales[js+1] - scales[js]))
    img2[j, :] = (np.sinh(sigma * (scales[js + 1] - s[j])) * K2[js, bs, :] + np.sinh(sigma * (s[j] - scales[js])) * K2[js + 1, bs, :]) \
             / np.sinh(sigma * (scales[js + 1] - scales[js]))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xg,yg = np.meshgrid(x, s)
surf = ax.plot_surface(xg, yg, img1, cmap = cm.hot, linewidth=0, antialiased=False)
ax.set_ylabel('Scales')
ax.set_xlabel('x')
plt.title(f'Kernel shapes at base scale {scales[bs]:.2f}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xg, yg, img2, cmap = cm.hot, linewidth=0, antialiased=False)
ax.set_ylabel('Scales')
ax.set_xlabel('x')
plt.title(f'Kernel shapes at base scale {scales[bs]:.2f} with positivity constraint')


#plt.show()


fout = h5py.File('MSKernel.h5', 'w')
fout.create_dataset('frequencies', data=freq)
fout.create_dataset('scales', data=scales)
fout.create_dataset('widths', data=1/np.sqrt(2*t))
fout.create_dataset('weights', data=weight)
fout.create_dataset('kappa', data=kappa)
#fout.create_dataset('positive_coefficients', data=kappa_pos)
fout.close()


