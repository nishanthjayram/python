"""
    Contains methods to test the autogradient implementation on a given set
    of points.

    Author  :   Nishanth Jayram (https://github.com/njayram44)
    Date    :   January 6, 2021
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# We initialize a set of points to test our autogradient.
points = [
    (-2, 2.7),
    (-1, 3),
    (0, 1.3),
    (1, 2.4),
    (3, 5.5),
    (4, 6.2),
    (5, 9.1),
]
va = V(0.)
vb = V(0.)
vc = V(0.)
vx = V(0.)
vy = V(0.)

def plot_points(points):
    """Plots and generates a pictorial representation of a set of points."""
    fig, ax = plt.subplots()
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'r+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_points_and_y(points, vx, oy):
    """Plots and generates a pictorial representation of the fitted curve."""
    fig, ax = plt.subplots()
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'r+')
    x_min, x_max = np.min(xs), np.max(xs)
    step = (x_max - x_min) / 100
    x_list = list(np.arange(x_min, x_max + step, step))
    y_list = []
    for x in x_list:
        vx.assign(x)
        oy.compute()
        y_list.append(oy.value)
    ax.plot(x_list, y_list)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def fit(loss, points, params, delta=0.0001, num_iterations=4000):
    """Determines a curve to fit a given set of points."""
    for iteration_idx in range(num_iterations):
        total_loss = 0.
        loss.zero_gradient()
        # We assign our point values to x and y before computing the loss
        # and backpropagating by computing its gradient; then, we append
        # the loss value to the total loss.
        for x, y in points:
            vx.assign(x)
            vy.assign(y)
            loss.compute()
            loss.compute_gradient()
            total_loss += loss.value
        if (iteration_idx + 1) % 100 == 0:
            print("Loss:", total_loss)
        for vv in params:
            vv.assign(vv.value - delta * vv.gradient)
    return total_loss

def main():
    matplotlib.rcParams['figure.figsize'] = (8.0, 3.)
    params = {'legend.fontsize': 'large',
            'axes.labelsize': 'large',
            'axes.titlesize':'large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large'}
    matplotlib.rcParams.update(params)

    plot_points(points)
    oy = va * vx * vx + vb * vx + vc
    loss = (vy - oy) * (vy - oy)
    lv = fit(loss, points, [va, vb, vc])
    plot_points_and_y(points, vx, oy)

if __name__ == "__main__":
    main()