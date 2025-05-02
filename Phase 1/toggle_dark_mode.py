import matplotlib.pyplot as plt

def toggle_dark_mode(ax=None, fig=None):
    """
    Toggles dark mode for a given Matplotlib axis or figure.
    
    Parameters:
        ax (matplotlib.axes.Axes, optional): The specific axis to toggle dark mode on.
        fig (matplotlib.figure.Figure, optional): The specific figure to toggle dark mode on.
            If neither ax nor fig is provided, the current figure is used.
    """
    if ax is None and fig is None:
        fig = plt.gcf()
    
    if fig is not None:
        axes = fig.get_axes()
    elif ax is not None:
        axes = [ax]
    else:
        axes = []

    for ax in axes:
        # Initialize the _is_dark_mode attribute if it doesn't exist
        if not hasattr(ax, '_is_dark_mode'):
            ax._is_dark_mode = False  # Default to light mode
        print(f"Current dark mode state: {ax._is_dark_mode}")
        if ax._is_dark_mode:
            # Revert to light mode
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            for text in ax.texts:
                text.set_color('black')
            if ax.title:
                ax.title.set_color('black')
            if ax.xaxis.label:
                ax.xaxis.label.set_color('black')
            if ax.yaxis.label:
                ax.yaxis.label.set_color('black')
            ax._is_dark_mode = False  # Update the state
        else:
            # Apply dark mode
            ax.set_facecolor('#121212')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            for text in ax.texts:
                text.set_color('white')
            if ax.title:
                ax.title.set_color('white')
            if ax.xaxis.label:
                ax.xaxis.label.set_color('white')
            if ax.yaxis.label:
                ax.yaxis.label.set_color('white')
            ax._is_dark_mode = True  # Update the state

    # Update the figure background color
    if fig is not None:
        if hasattr(axes[0], '_is_dark_mode') and axes[0]._is_dark_mode:
            fig.patch.set_facecolor('#121212')  # Dark background for the figure
        else:
            fig.patch.set_facecolor('white')  # Light background for the figure
        fig.canvas.draw_idle()