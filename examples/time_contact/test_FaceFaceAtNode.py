import sys
sys.path.append('.')
from Mortar2 import *
from jax import jit
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

N = 100
S = np.linspace(0.0, 1.0, N)

@jit
def create_surface_0(t):
    #x0a = np.array([0.0, 0.0])
    #x0b = np.array([1.0, -S[t]])
    #x0c = np.array([1.2, 0.5])
    x0a = np.array([0.0+S[t], 0.0])
    x0b = np.array([0.2+S[t], -.1])
    x0c = np.array([0.4+S[t], 0.0])
    return np.array([x0a, x0b, x0c])

@jit
def create_surface_1(t):
    #x1a = np.array([0.11, -0.7])
    #x1b = np.array([0.8, -0.2 + 0.5*S[t]])
    #x1c = np.array([-0.1, -0.4])
    x1a = np.array([1.0, -0.2])
    x1b = np.array([0.3, -0.2])
    x1c = np.array([-1.0, -0.2])
    return np.array([x1a, x1b, x1c])


nodal_gaps = np.zeros((N,6))
nodal_areas = np.zeros((N,6))

for i_s, s in enumerate(S):
    surface0 = create_surface_0(i_s)
    conn0 = np.array([[0,1],[1,2]]) #,[2,0]])
    surface0 = jax.vmap(lambda x,conn: x[conn], (None,0))(surface0, conn0)

    surface1 = create_surface_1(i_s)
    conn1 = np.array([[0,1],[1,2]])
    surface1 = jax.vmap(lambda x,conn: x[conn], (None,0))(surface1, conn1)
    conn1 = conn1+3

    for edge0,c0 in zip(surface0, conn0):
        for edge1,c1 in zip(surface1, conn1):
            integrals0, integrals1, areas0, areas1 = integrate_gap_against_shape(edge0, edge1)
            nodal_gaps = nodal_gaps.at[i_s,c0].add(integrals0)
            nodal_gaps = nodal_gaps.at[i_s,c1].add(integrals1)
            nodal_areas = nodal_areas.at[i_s,c0].add(areas0)
            nodal_areas = nodal_areas.at[i_s,c1].add(areas1)

    #print('nodal gaps = ', nodal_gaps)
    #print('nodal areas = ', nodal_areas)

nodal_areas = nodal_areas.at[nodal_areas==0].set(1.0)
nodal_gaps = nodal_gaps / nodal_areas

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
    #for e,edge in enumerate(surface):
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
n2 = 4
curve0, = axs[1].plot(S,  nodal_gaps[:,n1], 'b')
dot0, = axs[1].plot(S[0], nodal_gaps[0,n1],'ro')
curve1, = axs[1].plot(S,  nodal_gaps[:,n2], 'g--')
dot1, = axs[1].plot(S[0], nodal_gaps[0,n2],'r*')
axs[1].plot(S, 0*S, '--k')

def update(val):
    t = t_slider.val
    redraw_surface(lines0, 0, t)
    redraw_surface(lines1, 1, t)

    dot0.set_xdata(S[t])
    dot0.set_ydata(nodal_gaps[t,n1])

    dot1.set_xdata(S[t])
    dot1.set_ydata(nodal_gaps[t,n2])

t_slider.on_changed(update)

plt.show()


