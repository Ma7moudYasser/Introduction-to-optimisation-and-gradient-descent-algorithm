#!/usr/bin/env python
# coding: utf-8

# # Notebook imports and packages

# In[253]:


import matplotlib.pyplot as plt
import numpy as np 

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm #color map

from sympy import symbols,diff
from math import log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# # Example 1 simple cost function 
# ## $$f(x) = x^2+x+1$$

# In[254]:


def f(x): 
    return x**2+x+1


# In[255]:


#Make Data

x_1 = np.linspace(start=-3, stop=3 , num=500)


# In[256]:


#plot function and derivative side by side
plt.figure(figsize=[15,5])

#1 Chart: cost function
plt.subplot(1,2,1)

plt.xlim(-3,3)
plt.ylim(0,8)

plt.title("cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("F(x)", fontsize=16)
plt.plot(x_1, f(x_1), color='blue', linewidth=3)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title("the slope of the cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("dF(x)", fontsize=16)
plt.grid()
plt.xlim(-2,3)
plt.ylim(-3,6)

plt.plot(x_1,df(x_1), color='skyblue', linewidth=5)
plt.show()


plt.show()


# ## Slope & Derivatives 

# In[257]:


def df(x): 
    return 2*x+1


# ## Python loops & Gradient Descent

# In[258]:


#Gradient Descent

new_x = 3
previous_x = 0
step_multiplier = 0.1
precision = 0.00001

x_list = [new_x]
slope_list = [df(new_x)]

for n in range(500):
    previous_x = new_x
    gradient= df(previous_x)
    new_x = previous_x - step_multiplier * gradient
    step_size = abs(new_x - previous_x)
    
    x_list.append(new_x)
    slope_list.append(df(new_x))
    if step_size < precision:
        print("The number of loops to reach precision is: ", n)
        break

print("Local minimum occurs at: ", new_x)
print("Slope or df(x) value at this point is: ", df(new_x))
print("F(x) value at this point is: ", f(new_x))
    
    


# In[259]:


#Superimpose the gradient descent calculations
plt.figure(figsize=[20,5])

#1 Chart: cost function
plt.subplot(1,3,1)

plt.xlim(-3,3)
plt.ylim(0,8)

plt.title("cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("F(x)", fontsize=16)
plt.plot(x_1, f(x_1), color='blue', linewidth=3)

values = np.array(x_list)
plt.scatter(x_list, f(values), color='red', s=100, alpha=0.8)

#2 Chart: Derivative
plt.subplot(1,3,2)

plt.title("the slope of the cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("dF(x)", fontsize=16)
plt.grid()
plt.xlim(-2,3)
plt.ylim(-3,6)

plt.plot(x_1,df(x_1), color='skyblue', linewidth=5)
plt.scatter(x_list, slope_list, color='red', s=100, alpha=0.6)




#3 Chart: Derivative(close up)
plt.subplot(1,3,3)

plt.title("Gradient Descent (close up)", fontsize=17)
plt.xlabel("X", fontsize=16)
plt.grid()
plt.xlim(-0.55,0.2)
plt.ylim(-0.3,0.8)

plt.plot(x_1,df(x_1), color='skyblue', linewidth=6, alpha=0.8)
plt.scatter(x_list, slope_list, color='red', s=300, alpha=0.6)

plt.show()


# # Example 2 - Multiple Minima vs Initial Guess & Advanced Functions
# 
# ## $$g(x) = x^4 - 4x^2 +5$$

# In[260]:


# Make some data 
x_2 = np.linspace(-2, 2, 1000)
print(x_2)


# In[261]:


def g(x):
    return x**4 - 4*x**2 + 5

def dg(x): 
    return 4*x**3 - 8*x 


# In[262]:


#plot function and derivative side by side
plt.figure(figsize=[15,5])

#1 Chart: cost function
plt.subplot(1,2,1)

plt.xlim(-2,2)
plt.ylim(0.5,5.5)

plt.title("cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("g(x)", fontsize=16)
plt.plot(x_2, g(x_2), color='blue', linewidth=3)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title("the slope of the cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("dg(x)", fontsize=16)
plt.grid()
plt.xlim(-2,3)
plt.ylim(-6,8)

plt.plot(x_2,dg(x_2), color='skyblue', linewidth=5)
plt.show()


# ## Gradient Descent as a python function

# In[263]:


#Gradient Descent
def gradient_descent(derivative_func, initial_guess, learning_rate= 0.02, precision = 0.001,
                    max_iter = 300):
    new_x = initial_guess


    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_iter):
        previous_x = new_x
        gradient= derivative_func(previous_x)
        new_x = previous_x - learning_rate * gradient
        step_size = abs(new_x - previous_x)

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))
        if step_size < precision:
            break
    return new_x, x_list, slope_list
    
    


# In[264]:


local_min, list_x, deriv_list = gradient_descent(dg, 0.5, 0.02, 0.001)
print('local Minimum occurs at: ', local_min)
print('Number of steps: ', len(list_x))


# In[265]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess = -0.5,
                                                 learning_rate=0.01, precision= 0.0001)
print('local Minimum occurs at: ', local_min)
print('Number of steps: ', len(list_x))


# In[266]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess = -0.1)
print('local Minimum occurs at: ', local_min)
print('Number of steps: ', len(list_x))


# In[267]:


#calling gradient Descent function

local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess = 0.1)
#plot function and derivative and scatter plot side by side
plt.figure(figsize=[15,5])

#1 Chart: cost function
plt.subplot(1,2,1)

plt.xlim(-2,2)
plt.ylim(0.5,5.5)

plt.title("cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("g(x)", fontsize=16)
plt.plot(x_2, g(x_2), color='blue', linewidth=3, alpha=0.8)
plt.scatter(list_x,g(np.array(list_x)), color= "red", s=100, alpha=0.6)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title("the slope of the cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("dg(x)", fontsize=16)
plt.grid()
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.plot(x_2,dg(x_2), color='skyblue', linewidth=5, alpha=0.6)
plt.scatter(list_x,deriv_list, color="red", s= 100, alpha=0.5)


plt.show()


# # Example 3- Divergence, overfolow and python tubles 
# ## $$h(x) = x^5 -2x^4 +2$$

# In[268]:


#Make data 

x_3 = np.linspace(start=-2.5, stop=2.5, num=1000)

#Basic Function 

def h(x):
    return x**5 - 2*x**4 + 2

def dh(x):
    return 5*x**4 - 8*x**3 


# In[269]:


#calling gradient Descent function

local_min, list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess = -0.2,
                                                max_iter=70)
#plot function and derivative and scatter plot side by side
plt.figure(figsize=[15,5])

#1 Chart: cost function
plt.subplot(1,2,1)

plt.xlim(-1.2,2.5)
plt.ylim(-1,4)

plt.title("cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("h(x)", fontsize=16)
plt.plot(x_3, h(x_3), color='blue', linewidth=3, alpha=0.8)
plt.scatter(list_x,h(np.array(list_x)), color= "red", s=100, alpha=0.6)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title("the slope of the cost function", fontsize=15)
plt.xlabel("X", fontsize=16)
plt.ylabel("dh(x)", fontsize=16)
plt.grid()
plt.xlim(-1,2)
plt.ylim(-4,5)

plt.plot(x_3,dh(x_3), color='skyblue', linewidth=5, alpha=0.6)
plt.scatter(list_x,deriv_list, color="red", s= 100, alpha=0.5)


plt.show()

print("local minimum occurs at: ", local_min)
print("Cost at this minimum is: ", h(local_min))
print("Number of steps: ", len(list_x))


# In[270]:


import sys
sys.float_info.max


# In[271]:


data_tuple = gradient_descent(derivative_func= dh , initial_guess= 0.2)
print("Local min is : ", data_tuple[0])


# ## The learning rate

# In[272]:


#Run gradient descent 3 times 

n = 100

low_gamma = gradient_descent(derivative_func=dg, initial_guess = 3,learning_rate = 0.0005, 
                             precision = 0.0001, max_iter= n)

mid_gamma = gradient_descent(derivative_func=dg, initial_guess = 3,learning_rate = 0.001, 
                             precision = 0.0001, max_iter= n)

high_gamma = gradient_descent(derivative_func=dg, initial_guess = 3,learning_rate = 0.002, 
                             precision = 0.0001, max_iter= n)

insane_gamma = gradient_descent(derivative_func=dg, initial_guess = 1.9, learning_rate=0.25,
                                precision = 0.0001, max_iter= n)

#plotting reduction in cost for each iteration
plt.figure(figsize=[20,10])

#1 Chart: cost function

plt.xlim(0,n)
plt.ylim(0,50)

plt.title("Effect of the learning rate", fontsize=15)
plt.xlabel("Number of iterations", fontsize=16)
plt.ylabel("Cost", fontsize=16)

#Values of our charts 
# 1) Y Axis Data: convert the lists to numpy arrays 

low_values = np.array(low_gamma[1])


# 2) X Axis Data: creat a list from 0 to n+1
iteration_list = list(range(0,n+1))

#Plotting low learning rate
plt.plot(iteration_list, g(low_values), color='lightgreen', linewidth=5)
plt.scatter(iteration_list, g(low_values), color = 'lightgreen', s=80)


#Plotting mid learning rate
plt.plot(iteration_list, g(np.array(mid_gamma[1])), color='steelblue', linewidth=5)
plt.scatter(iteration_list, g(np.array(mid_gamma[1])), color = 'steelblue', s=80)

#Plotting high learning rate
plt.plot(iteration_list, g(np.array(high_gamma[1])), color='hotpink', linewidth=5)
plt.scatter(iteration_list, g(np.array(high_gamma[1])), color = 'hotpink', s=80)

#Plotting insane learning rate
plt.plot(iteration_list, g(np.array(insane_gamma[1])), color= 'red', linewidth=5)
plt.scatter(iteration_list, g(np.array(insane_gamma[1])), color= 'red', s=80)

plt.show()


# # Example 4- Data vis with 3D charts 
# ## Minimise $$f(x,y) = \frac{1}{3^{-x^2-y^3} + 1}$$
#  Minimise  $$f(x,y) = \frac{1}{r}$$ where $r$ is $3^{-x^2-y^2}$

# In[273]:


def f(x,y):
    r = 3**(-x**2-y**2)
    return 1/(r+1)


# In[274]:


#Make our data

x_4 = np.linspace(start=-2, stop=2, num=200)
y_4 = np.linspace(start=-2, stop=2, num=200)


print("shape of X array ", x_4.shape)


x_4, y_4 = np.meshgrid(x_4, y_4)

print("shape after meshgrid: ", x_4.shape)


# In[275]:


#Generating 3D plot 

fig= plt.figure(figsize=[16,12])
#ax= fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x,y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap = cm.coolwarm, alpha=0.4)
#gca is an abbreviation for get current axis
plt.show()


# # Partial Derivative & Symbolic computation
# ## $$\frac{\partial f}{\partial x} = \frac {2x \ln(3) \cdot 3^{-x^2 - y^2}}{\left(3^{-x^2 - y^2} + 1\right)^2} $$
# 
# ## $$\frac{\partial f}{\partial y} = \frac {2y \ln(3) \cdot 3^{-x^2 - y^2}}{\left(3^{-x^2 - y^2} + 1\right)^2} $$

# In[276]:


a, b = symbols('x, y')
print('Our cost function f(x,y) is: ',f(a, b))
print('Our partial derivative with respect to x: ', diff(f(a,b), a))
print("Value of f(x,y) at values x=1.8 and y = 1.0 is: ",
      f(a,b).evalf(subs={a:1.9,b:1.0})) #python dictionary
print("The value of Partial derivative with respect to x at x = 1.8 and y = 1.0 is: ", diff(f(a,b), a).evalf(subs={a:1.8, b:1.0}))


# In[277]:


#Partial derivatives example 4

def fpx(x,y):
    r= 3**(-x**2 - y**2)
    return 2*x*log(3)*r/(r+1)**2

def fpy(x,y):
    r= 3**(-x**2 - y**2)
    return 2*y*log(3)*r/(r+1)**2


# In[278]:


fpx(1.8, 1.0)


# ## Batch Gradienct Descent with SymPy

# In[279]:


#setup 
multiplier = 0.1
max_iter = 500
parameters = np.array([1.8, 1.0]) #initial guess


for n in range(max_iter):
    gradient_x = fpx(parameters[0], parameters[1])
    gradient_y = fpy(parameters[0], parameters[1])
    gradients = np.array([gradient_x, gradient_y])
    parameters = parameters - multiplier * gradients 

    
# Results 

print("Values in gradient array: ", gradients)
print("The minimum value of x is: ", parameters[0])
print("The minimum value of y is: ", parameters[1])
print("The cost is: ", f(parameters[0], parameters[1]))
    


# # Graphing 3D gradient descent and advanced numpy arrays 

# In[280]:


#setup 
multiplier = 0.1
max_iter = 500
parameters = np.array([1.8, 1.0]) #initial guess

values_array = parameters.reshape(1,2)


for n in range(max_iter):
    gradient_x = fpx(parameters[0], parameters[1])
    gradient_y = fpy(parameters[0], parameters[1])
    gradients = np.array([gradient_x, gradient_y])
    parameters = parameters - multiplier * gradients 
    #values_array = np.append(values_array, parameters.reshape(1,2), axis=0)
    values_array = np.concatenate((values_array, parameters.reshape(1,2)), axis=0)
# Results 

print("Values in gradient array: ", gradients)
print("The minimum value of x is: ", parameters[0])
print("The minimum value of y is: ", parameters[1])
print("The cost is: ", f(parameters[0], parameters[1]))
    


# In[281]:


#Generating 3D plot 

fig= plt.figure(figsize=[16,12])
#ax= fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x,y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap = cm.coolwarm, alpha=0.4)
ax.scatter(values_array[:, 0], values_array[:, 1],
          f(values_array[:, 0], values_array[:, 1]), 
           s=50, color = "red")
#gca is an abbreviation for get current axis
plt.show()


# In[282]:


# Advanced Numpy array practice 
kirk = np.array([["Murad", "Amgad"]])


print(kirk.shape)
mariam = np.array([["Captain", "Guitar"], ["mahmoud", "yasser"]])
print(mariam.shape)

print(mariam[1][1])

the_roots = np.append(arr= mariam, values= kirk, axis=0)
print(the_roots)

# How to print only the first column in the matrix

print("The first column data is,.....", the_roots[:, 0]) # the : notation stands for selecting all the rows in the table 
the_roots = np.append(arr=the_roots, values=[["Malik b", "Mazen"]], axis=0)
print("The roots data of the course ..... ", the_roots[:,:])
print("The roots data of the course ..... ", the_roots[:,1])


# # Example 5- Working with data and real cost functions 
# ## Mean square error: A cost function for regression problems
# ### $$RSS = \sum_{i=1}^{n} \big (y^{(i)} - h_\theta x^{(i)} \big )^2$$
# ### $$MSE = \frac{1}{n} \sum_{i=1}^{n} \big (y^{(i)} - h_\theta x^{(i)} \big )^2$$
# ### $$MSE = \frac{1}{n} \sum_{i=1}^{n} \big (y - \hat{y} \big )^2$$

# In[283]:


# Make sample data 

x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5= np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7,1) #transpose and reshape both make the same function here
print("Shape of x_5 array: ", x_5.shape)
print("Shape of y_5 array: ", y_5.shape)


# In[284]:


# Quick linear regression
regr = LinearRegression()
regr.fit(x_5, y_5)
print("Theta 0 :", regr.intercept_[0])
print("Theta 1 :", regr.coef_[0][0])


# In[285]:


plt.scatter(x_5, y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color = "orange", linewidth=3)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()


# In[286]:


y_hat = 0.8475351486029536 + 1.2227264637835915*x_5
print("Estimated values y_hat are: \n", y_hat)
print("In comparison , the actual y values are: \n", y_5)


# In[287]:


def mse(y,y_hat):
    #mse_calc = 1/(y.size) * sum((y-y_hat)**2)
    mse_calc = np.average((y-y_hat)**2, axis=0)
    return mse_calc


# In[288]:


print("manually calculated mse",mse(y_5, y_hat))
print("mse regression using automatic function in sklearn: ", mean_squared_error(y_5, y_hat))
print("mse regression is ", mean_squared_error(y_5, regr.predict(x_5)))


# ## 3D plot for the MSE cost function
# ### Make data for thetas

# In[298]:


nr_thetas = 200
th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)
plot_t1


# ## Calc mse using nested loops

# In[299]:


plot_cost = np.zeros((nr_thetas,nr_thetas))
plot_cost


for i in range(nr_thetas):
    for j in range(nr_thetas):
        #print(plot_t0[i][j])
        y_hat = plot_t0[i][j] + plot_t1[i][j]*x_5
        plot_cost[i][j] = mse(y_5, y_hat)
print("Shape of plot_t0", plot_t0.shape)
print("Shape of plot_t1", plot_t1.shape)
print("Shape of plot_cost", plot_cost.shape)


# In[301]:


# Plotting MSE

fig = plt.figure(figsize = [16,12])
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection = '3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap= cm.hot)
plt.show()


# In[307]:


print("Min value of plot_cost", plot_cost.min())
ij_min = np.unravel_index(indices= plot_cost.argmin(), shape = plot_cost.shape)
print('Min occurs at ij_min', ij_min)
print('Min MSE for Theta 0 at plot_t0 [111][91]', plot_t0[111][91])
print('Min MSE for Theta 1 at plot_t0 [111][91]', plot_t1[111][91])


# ### Partial derviative of MSE with respect to $\theta_0$ and $\theta_1$
# ## $$\frac{\partial MSE}{\partial \theta_0} = - \frac{2}{n} \sum_{i=1}^{n} \big (y^{(i)} -
# \theta_0 - \theta_1  x^{(i)} \big)$$
# ## $$\frac{\partial MSE}{\partial \theta_1} = - \frac{2}{n} \sum_{i=1}^{n} \big (y^{(i)} -
# \theta_0 - \theta_1  x^{(i)} \big) \big(x^{(i)} \big)$$

# ## MSE && Gradient Descent 

# In[308]:


# x values, y values, array of theta parameters (theta0 at index 0 and theta1 at index 1)
def grads(x,y,thetas):
    n= y.size
    
    theta0_slope = -2/n * sum(y-thetas[0]-thetas[1]*x)
    theta1_slope = -2/n * sum((y-thetas[0] - thetas[1]*x)*(x))
    
    
    #return np.array([theta0_slope[0], theta1_slope[1]])
    #return np.append(arr= theta0_slope, values= theta1_slope)
    return np.concatenate((theta0_slope, theta1_slope), axis=0)


# In[310]:


multiplier = 0.01
thetas = np.array([2.9, 2.9])

#collect data points for scatter plot
plot_vals = thetas.reshape(1,2)
mse_vals = mse(y_5, thetas[0] + thetas[1]*x_5)
for i in range(1000):
    thetas = thetas - multiplier* grads(x_5, y_5, thetas)
    
    #Append the new values to our numpy array 
    plot_vals =  np.concatenate((plot_vals, thetas.reshape(1,2)), axis = 0)
    mse_vals = np.append(arr= mse_vals, values= mse(y_5, thetas[0]+ thetas[1]*x_5))
    
#Results
print("Min occurs at theta 0: ",thetas[0])
print("Min occurs at theta 1: ",thetas[1])
print("MSE is: ", mse(y_5, thetas[0] + thetas[1]*x_5))


# In[312]:


# Plotting MSE

fig = plt.figure(figsize = [16,12])
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection = '3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.scatter(plot_vals[:, 0], plot_vals[:, 1], mse_vals, s=80, color='black')

ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap= cm.rainbow, alpha= 0.4)
plt.show()


# In[ ]:





# In[291]:


# Nested loops 

for i in range(3):
    for j in range(3):
        print(f'value of is is {i}, value of j is {j}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




