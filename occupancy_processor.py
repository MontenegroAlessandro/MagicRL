import json
import numpy as np
import matplotlib.pyplot as plt
import os

class OccupancyProcessor:
    def __init__(self):
        self.positions = []
        self.save_directory = "/Users/leonardo/Desktop/Thesis/MagicRL/images"

    def add_position(self, position):
        self.positions.append(position)

    def plot_occupancy(self):

        print(self.positions)
        # Extract x and y coordinates
        x_coords = np.array([point['p'][0] for point in self.positions])
        y_coords = np.array([point['p'][1] for point in self.positions])

        # Define grid parameters
        grid_size = 7  # 7x7 grid
        step = 0.1

        # Create 2D histogram with bins based on the step and grid size
        x_bins = np.arange(0, grid_size + step, step)
        y_bins = np.arange(0, grid_size + step, step)

        # Calculate occupancy counts for each bin
        occupancy_counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])

        # Plotting the occupancy data
        plt.figure(figsize=(8, 8))
        plt.imshow(
            occupancy_counts.T,
            origin='lower',
            cmap='hot',
            interpolation='nearest',
            extent=[0, grid_size, 0, grid_size]
        )
        plt.colorbar(label='Occupancy Count')
        plt.xlabel('X Coordinate (binned)')
        plt.ylabel('Y Coordinate (binned)')
        plt.title('Occupancy Plot in 7x7 Grid')
        plt.xticks(np.arange(0, grid_size + 1, 1))
        plt.yticks(np.arange(0, grid_size + 1, 1))
        plt.grid(False)
        save_path = os.path.join(self.save_directory, "occupancy_plot.png")
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved to {save_path}")

def main():
    # Example usage
    occupancy_processor = OccupancyProcessor()
    # Add positions as required, e.g.,
    occupancy_processor.add_position({'p': [2.4, 5.7]})
    occupancy_processor.add_position({'p': [2.8, 7.0]})
    occupancy_processor.add_position({'p': [1.1, 7.0]})
    occupancy_processor.add_position({'p': [0.0, 2.4]})
    occupancy_processor.add_position({'p': [2.4, 5.8]})
    # Plot the data
    occupancy_processor.plot_occupancy()


if __name__ == "__main__":
    main()

