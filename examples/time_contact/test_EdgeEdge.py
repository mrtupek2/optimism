import sys
sys.path.append('.')
from Mortar2 import *
from optimism.JaxConfig import *
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios' : [1,1.5]})

line, = axs[1].plot(S, G)
dot, = axs[1].plot([S[0]], [G[0]], 'or')

edgeline0, = axs[0].plot(edges0[0,:,0], edges0[0,:,1], 'k')
edgeline1, = axs[0].plot(edges1[0,:,0], edges1[1,:,1], 'k')

cpline0, = axs[0].plot(cutpoints0[0,:,0], cutpoints0[0,:,1], 'gv')
cpline1, = axs[0].plot(cutpoints1[0,:,0], cutpoints1[0,:,1], 'gv')

axs[0].set_xlim([-0.5, 1.5])
axs[0].set_ylim([-1.4, 0.6])
axs[0].set_aspect('equal')
axs[1].set_box_aspect(0.5)

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
