import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
num_steps = 1000
alpha = 1.5
scale = 1.0

x_trajectory, y_trajectory = levy_flight(num_steps, alpha, scale)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize the plot
ax.set_xlim(x_trajectory.min(), x_trajectory.max())
ax.set_ylim(y_trajectory.min(), y_trajectory.max())
ax.set_title("Lévy Flight Trajectory")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid(True)

# Initialize the line
line, = ax.plot([], [], color='blue', alpha=0.5)

# Initialize the scatter plot
scatter = ax.scatter([], [], color='black')

# Function to update the plot
def update(i):
    line.set_data(x_trajectory[:i+1], y_trajectory[:i+1])
    scatter.set_offsets([x_trajectory[i], y_trajectory[i]])
    return line, scatter,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)

# Save the animation as a GIF file
ani.save('levy_flight.gif', writer='pillow', fps=30)
