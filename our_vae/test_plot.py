import matplotlib.pyplot as plt
import numpy as np

max_lr = 0.01
min_lr = 10e-6
peak_lr_it = 50
slope_lr_beginning = (max_lr - min_lr)/peak_lr_it
stop_decline_it = 100 
slope_lr_ending = (min_lr - max_lr)/(stop_decline_it-peak_lr_it)

def learning_rate_func(it):
    """
    it: the global iteration
    """
    if it < peak_lr_it:
        return slope_lr_beginning*it
    else:
        it = it-peak_lr_it+1
        return max(slope_lr_ending*it+max_lr, min_lr)

plotting_max = 200
X = np.linspace(0, plotting_max, plotting_max)
Y = [learning_rate_func(x) for x in X]
plt.plot(X, Y)
plt.show()
