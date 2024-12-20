import numpy as np
import matplotlib.pyplot as plt

def levy_flight(num_steps, alpha=1.5, scale=1.0):
    """
    Generate a Lévy flight in 2D space.

    Parameters:
    num_steps (int): The number of steps in the Lévy flight.
    alpha (float, optional): The exponent of the power-law distribution. Defaults to 1.5.
    scale (float, optional): The scale of the power-law distribution. Defaults to 1.0.

    Returns:
    numpy.ndarray: The Lévy flight trajectory.
    """
    # Generate the step lengths from a power-law distribution
    steps = np.random.pareto(alpha, num_steps) * scale

    # Generate the directions of the steps
    directions = np.random.uniform(0, 2*np.pi, num_steps)

    # Calculate the x and y components of the steps
    x_steps = steps * np.cos(directions)
    y_steps = steps * np.sin(directions)

    # Calculate the Lévy flight trajectory
    x_trajectory = np.cumsum(x_steps)
    y_trajectory = np.cumsum(y_steps)

    return x_trajectory, y_trajectory

# Generate a Lévy flight
num_steps = 2000
alpha = 1.5
scale = 1

x_trajectory, y_trajectory = levy_flight(num_steps, alpha, scale)

# Plot the Lévy flight trajectory
plt.figure(figsize=(8, 8))

# Plot the entire trajectory in blue
plt.plot(x_trajectory, y_trajectory, color='blue', alpha=0.5, label='Entire Trajectory')

# Plot the first 100 steps in red
plt.plot(x_trajectory[:100], y_trajectory[:100], color='red', label='First 100 Steps')

# Plot the last 100 steps in green
plt.plot(x_trajectory[-100:], y_trajectory[-100:], color='green', label='Last 100 Steps')

# Plot the starting point in black
plt.scatter(x_trajectory[0], y_trajectory[0], color='black', label='Starting Point')

# Plot the ending point in yellow
plt.scatter(x_trajectory[-1], y_trajectory[-1], color='yellow', label='Ending Point')

plt.title("Lévy Flight Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()
plt.show()
