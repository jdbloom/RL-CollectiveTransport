import numpy as np
import matplotlib.pyplot as plt

def create_spiral_plot(angles, times=None, figsize=(10, 10)):
    """
    Create a spiral plot for rotational data.
    
    Parameters:
    angles : array-like
        The rotation angles in degrees (0-360)
    times : array-like, optional
        The time points for each angle. If None, will use indices.
    figsize : tuple, optional
        Figure size in inches (width, height)
    """
    # If no times provided, use indices
    if times is None:
        times = np.arange(len(angles))
    
    # Convert angles to radians
    theta = np.deg2rad(angles)
    
    # Create radius that increases with time
    # Normalize time to create a nice spiral
    r = 1 + (times - min(times)) / (max(times) - min(times))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Plot the spiral
    ax.plot(theta, r, 'b-', linewidth=2)
    
    # Add points at each measurement
    ax.scatter(theta, r, c='red', s=50)
    
    # Customize the plot
    ax.set_rticks([])  # Remove radial ticks
    ax.set_rlim(0, max(r) + 0.1)  # Set radius limit with some padding
    
    # Set theta ticks to degrees
    ax.set_xticks(np.pi/180. * np.linspace(0, 360, 8, endpoint=False))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
    
    plt.title('Rotational Data Spiral Plot')
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    n_points = 50
    times = np.linspace(0, 10, n_points)
    # Create sample angles that progress with some noise
    angles = (times * 45 + np.random.normal(0, 10, n_points)) % 360
    
    # Create the plot
    fig, ax = create_spiral_plot(angles, times)
    plt.show()