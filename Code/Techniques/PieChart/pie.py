import matplotlib.pyplot as plt
import numpy as np

def pie_chart(segmented_image: np.array, label_colours: np.array) -> np.array:
    
    """ Create pie chart of segmented regions and return it as a numpy array"""
    
    # Count (unique) segmented regions with their area
    regions, counts = np.unique(segmented_image, return_counts=True)
        
    # Extract their colours (and normalise to meet matplotlib criteria)
    colours = label_colours[regions]/255
    
    # Create figure
    fig = plt.figure(figsize=(5,5))
    
    # Plot pie chart (labels are integer for regions)
    plt.pie(counts, colors=colours, labels=regions)
    
    # Close plotting
    plt.close()
    
    # Convert chart to numpy array
    pie_numpy = figure_to_array(fig)
    
    return pie_numpy
        

def figure_to_array(fig):
    
    """ Convert matplotlib figure to numpy array """
    
    # Draw figure on canvas
    fig.canvas.draw()
    
    # Extract numpy representation
    return np.array(fig.canvas.renderer._renderer)