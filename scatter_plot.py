import matplotlib.pyplot as plt

# scatter plot of two input features x and y and output class labels 
def scatter_plot(x, y, output, sizes, plot_title):
    # Create a scatter plot using Matplotlib
    scatter = plt.scatter(x, y, c=output, s=sizes)

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(plot_title)

    # Create a legend using the scatter points and class labels
    plt.legend(*scatter.legend_elements(), title='Classes')
    # Display the plot
    plt.show()

    return