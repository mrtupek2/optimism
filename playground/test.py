import matplotlib.pyplot as plt
import numpy as np

print('hello world')

x = [3, 4]
print(x)

#loop from 0 to 9
def cbrt(i):
    return np.cbrt(i*100)

sum = 0
for i in range(10):
    sum+=cbrt(i)

#plot x^4 vs x
x = np.linspace(-10, 10, 100)
#plot the line with color blue and dashed line
plt.plot(x, x**4, 'b--')
#add a title
plt.title('x^4')
#add x and y labels
plt.xlabel('x')
plt.ylabel('x^4')

plt.show()

#find the index with the largest values
x = np.array([1, 2, 3, 4, 5])
index = np.argmax(x)
print(index)

#write a loop that does nothing
for i in range(10):
    pass

def f(x):
    return x**2

#take the derivative of f(x)
#import jax.numpy as jnp
#from jax import grad
#grad_f = grad(f)
#print(grad_f(2.0))

print(x)


#compute the shortest path in a graph
import networkx as nx
G = nx.Graph()
G.add_edge(1, 2, weight=4)
G.add_edge(1, 3, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(2, 4, weight=2)
G.add_edge(3, 4, weight=5)
G.add_edge(3, 5, weight=10)
G.add_edge(4, 5, weight=1)
path = nx.dijkstra_path(G, 1, 5)

#plot the graph
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

