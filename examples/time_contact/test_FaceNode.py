import sys
sys.path.append('.')
from Mortar2 import *
from jax import jit
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

N = 200
S = np.linspace(-0.2, 0.5, N)

@jit
def create_surface_0(t):
    x0a = np.array([0.0+S[t], 0.0])
    x0b = np.array([0.2+S[t], -0.1])
    x0c = np.array([0.4+S[t], 0.0])
    return np.array([x0a, x0b, x0c])

@jit
def create_surface_1(t):
    x1a = np.array([1.0, -0.2])
    x1b = np.array([0.3, -0.2])
    x1c = np.array([-1.0, -0.2])
    return np.array([x1a, x1b, x1c])


def cpp(p, edge):
    #min norm(p - edge[0] * (1-xi) - edge[1] * xi)
    dp = p - edge[0]
    de = edge[1] - edge[0]
    xi = (dp @ de) / (de @ de)
    xi = np.maximum(np.minimum(xi, 1.0), 0.0)
    return np.linalg.norm(dp - xi * de)

def normal_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

def approx_cpp(p, edge, edge1):
    dp = p - edge[0]
    de = edge[1] - edge[0]

    xi = (dp @ de) / (de @ de)
    
    ref_edge_length = np.linalg.norm(edge1[1]-edge1[0])
    edge_length = np.linalg.norm(edge[1]-edge[0])

    if xi < 0.0: return np.linalg.norm(dp) * edge_length / ref_edge_length
    if xi > 1.0: return np.linalg.norm(dp-de) * edge_length / ref_edge_length

    return np.abs(normal_2d(dp, de)) / ref_edge_length

nodal_gaps = np.zeros((N,6))
approx_nodal_gaps = np.zeros((N,6))

conn1 = np.array([[0,1],[1,2]])
last_surface1 = create_surface_1(len(S)-1)
last_surface1 = jax.vmap(lambda x,conn: x[conn], (None,0))(last_surface1, conn1)

for i_s, s in enumerate(S):
    surface0 = create_surface_0(i_s)

    surface1 = create_surface_1(i_s)
    surface1 = jax.vmap(lambda x,conn: x[conn], (None,0))(surface1, conn1)

    for p,point in enumerate(surface0):
        for edge,last_edge,c1 in zip(surface1, last_surface1, conn1):
            dist = cpp(point, edge)
            nodal_gaps = nodal_gaps.at[i_s,p].add(dist)
            approx_dist = approx_cpp(point, edge, last_edge)
            approx_nodal_gaps = approx_nodal_gaps.at[i_s,p].add(approx_dist)

conn1 = conn1+3

def draw_surface(ax, num, t):
    surface = create_surface_0(t) if num==0 else create_surface_1(t)
    lines = []
    line, = ax.plot(surface[:,0], surface[:,1], 'k')
    leftdot, = ax.plot(surface[:,0], surface[:,1], 'ok')
    lines.append(line)
    lines.append(leftdot)
    return lines

def redraw_surface(lines, num, t):
    surface = create_surface_0(t) if num==0 else create_surface_1(t)
    lines[0].set_xdata(surface[:,0])
    lines[0].set_ydata(surface[:,1])
    lines[1].set_xdata(surface[:,0])
    lines[1].set_ydata(surface[:,1])

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios' : [1,1.5]})

lines0 = draw_surface(axs[0], 0, 0)
lines1 = draw_surface(axs[0], 1, 0)

axs[0].set_xlim([-1.2, 1.2])
axs[0].set_ylim([-1.2, 1.2])
axs[0].set_aspect('equal')
axs[1].set_box_aspect(0.5)

fig.subplots_adjust(bottom=0.25)
axn = fig.add_axes([0.25, 0.1, 0.65, 0.03])
allowed_amplitudes = np.arange(N)
t_slider = Slider(axn, 'time', 0, N, valinit=0, valstep = allowed_amplitudes)

n1 = 1
curve0, = axs[1].plot(S,  nodal_gaps[:,n1], 'b')
dot0, = axs[1].plot(S[0], nodal_gaps[0,n1],'ro')
curve1, = axs[1].plot(S,  approx_nodal_gaps[:,n1], 'g--')
dot1, = axs[1].plot(S[0], approx_nodal_gaps[0,n1],'r*')
axs[1].plot(S, 0*S, '--k')

def update(val):
    t = t_slider.val
    redraw_surface(lines0, 0, t)
    redraw_surface(lines1, 1, t)

    dot0.set_xdata(S[t])
    dot0.set_ydata(nodal_gaps[t,n1])

    dot1.set_xdata(S[t])
    dot1.set_ydata(approx_nodal_gaps[t,n1])

t_slider.on_changed(update)

plt.show()


