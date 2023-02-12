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
plt.plot(x, x**4)
#color the line blue, with dashed line
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