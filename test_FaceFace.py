import sys
sys.path.append('..')
from optimism.JaxConfig import *
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

import jax
import jax.numpy as np

def compute_normal(edge):
    tangent = edge[1]-edge[0]
    normal = np.array([tangent[1], -tangent[0]])
    return normal / np.linalg.norm(normal)


def average_normals(normal0, normal1):
    normalDiff = normal0 - normal1
    return normalDiff / np.linalg.norm(normalDiff)


def compute_average_normal(edgeA, edgeB):
    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    normal = nA - nB
    return normal / np.linalg.norm(normal)


def eval_linear_field_on_edge(field, xi):
    #print(field.shape, xi.shape)
    return field[0] * (1.0 - xi) + field[1] * xi


def compute_intersection(edgeA, edgeB, normal):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = np.array([edgeB[0]-edgeB[1], normal]).T
        r = np.array(edgeB[0]-xa)
        xig = np.linalg.solve(M,r)
        return xig[0], xig[1]

    xiBs1, gs1 = jax.vmap(compute_xi, (0,None,None))(edgeA, edgeB, normal)
    xiAs2, gs2 = jax.vmap(compute_xi, (0,None,None))(edgeB, edgeA,-normal)

    xiAs = np.hstack((np.arange(2), xiAs2))
    xiBs = np.hstack((xiBs1, np.arange(2)))
    gs = np.hstack((gs1, gs2))

    xiAgood = jax.vmap(lambda xia, xib: np.where((xia >= 0.0) & (xia <= 1.0) & (xib >= 0.0) & (xib <= 1.0), xia, np.nan))(xiAs, xiBs)
    argsMinMax = np.array([np.nanargmin(xiAgood), np.nanargmax(xiAgood)])

    return xiAs[argsMinMax], xiBs[argsMinMax], gs[argsMinMax]


def length(edge):
    return np.linalg.norm(edge[0]-edge[1])


@jax.jit
def integrate_gap(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)
    integralLength = (xi0[1]-xi0[0]) * length(edge0) * (n0 @ n)
    return 0.5 * (g[0]+g[1]) * integralLength


@jax.jit
def integrate_gap_against_shape(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)
    integralLength = (xi0[1]-xi0[0]) * length(edge0) * (n0 @ n)

    # do we need gauss lobatto [-1,0,1], [1/3, 4/3, 1/3] ?
    # I don't think so, we only care about integrating constants exactly..., but its funny

    Nl0 = 1.0 - xi0
    Nr0 = xi0
    left0 = 0.5 * (g @ Nl0) * integralLength
    right0 = 0.5 * (g @ Nr0) * integralLength

    Nl1 = 1.0 - xi1
    Nr1 = xi1
    left1 = 0.5 * (g @ Nl1) * integralLength
    right1 = 0.5 * (g @ Nr1) * integralLength

    return np.array([left0, right0]), np.array([left1, right1])


@jax.jit
def get_cut_coordinates(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)

    left0 = eval_linear_field_on_edge(edge0, xi0[0])
    right0 = eval_linear_field_on_edge(edge0, xi0[1])

    left1 = eval_linear_field_on_edge(edge1, xi1[0])
    right1 = eval_linear_field_on_edge(edge1, xi1[1])

    return np.array([left0, right0]), np.array([left1, right1])


x0a = np.array([0.0, 0.0])
x0b = np.array([1.0, 0.0])
edge0 = np.array([x0a, x0b])

x1a = np.array([0.8, -0.2])
x1b = np.array([-0.1, -0.4])
edge1 = np.array([x1a, x1b])

N = 200
S = np.linspace(-0.5, 2.0, N)
G = []

edges0 = []
edges1 = []

cutpoints0 = []
cutpoints1 = []

for s in S:
    edge0_mod = edge0
    edge1_mod = edge1

    #edge1_mod = edge1_mod.at[0,0].set(edge1[0][0] + s)
    #edge1_mod = edge1_mod.at[1,0].set(edge1[1][0] + s)
    edge1_mod = edge1_mod.at[1,1].set(edge1_mod[1][1] - 0.1*s)
    edge1_mod = edge1_mod.at[1,0].set(edge1_mod[1][0] + s)

    edges0.append(edge0_mod)
    edges1.append(edge1_mod)

    gIntegral = integrate_gap(edge0_mod, edge1_mod)
    G.append(gIntegral)

    # gs0, gs1 = integrate_gap_against_shape(edge0_mod, edge1_mod)
    cp0, cp1 = get_cut_coordinates(edge0_mod, edge1_mod)

    cutpoints0.append(cp0)
    cutpoints1.append(cp1)

G = np.array(G)
edges0 = np.array(edges0)
edges1 = np.array(edges1)
cutpoints0 = np.array(cutpoints0)
cutpoints1 = np.array(cutpoints1)

fig, axs = plt.subplots(2)

line, = axs[1].plot(S, G)
dot, = axs[1].plot([S[0]], [G[0]], 'or')

edgeline0, = axs[0].plot(edges0[0,:,0], edges0[0,:,1], 'k')
edgeline1, = axs[0].plot(edges1[0,:,0], edges1[1,:,1], 'k')

cpline0, = axs[0].plot(cutpoints0[0,:,0], cutpoints0[0,:,1], 'gv')
cpline1, = axs[0].plot(cutpoints1[0,:,0], cutpoints1[0,:,1], 'gv')

axs[0].set_xlim([-0.5, 1.5])
axs[0].set_ylim([-1.4, 0.6])
axs[0].set_aspect('equal')

fig.subplots_adjust(bottom=0.25)
axn = fig.add_axes([0.25, 0.1, 0.65, 0.03])
allowed_amplitudes = np.arange(N)
t_slider = Slider(axn, 'time', 0, N, valinit=0, valstep = allowed_amplitudes)

def draw_edge(edgeline, edgedata, cutline, cutdata, t):
    edgeline.set_xdata(edgedata[t,:,0])
    edgeline.set_ydata(edgedata[t,:,1])
    cutline.set_xdata(cutdata[t,:,0])
    cutline.set_ydata(cutdata[t,:,1])

# Update function
def update(val):
    t = t_slider.val

    line.set_xdata(S)
    line.set_ydata(G)
    dot.set_xdata([S[t]])
    dot.set_ydata([G[t]])

    draw_edge(edgeline0, edges0, cpline0, cutpoints0, t)
    draw_edge(edgeline1, edges1, cpline1, cutpoints1, t)

    #edgeline0.set_xdata(edges0[t,:,0])
    #edgeline0.set_ydata(edges0[t,:,1])
    #edgeline1.set_xdata(edges1[t,:,0])
    #edgeline1.set_ydata(edges1[t,:,1])

    fig.canvas.draw_idle()

# Connect the slider to the update function
t_slider.on_changed(update)

plt.show()

