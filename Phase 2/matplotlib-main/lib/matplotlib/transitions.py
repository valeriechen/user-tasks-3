"""
Smooth transitions between different states of matplotlib plots.

This module provides functions to create smooth animations transitioning
between different states of a plot, supporting various plot types and
transition properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.artist import Artist
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Union, Callable, Optional, Any, Iterable
import copy


# Easing functions for smooth transitions
def linear_ease(t: float) -> float:
    """Linear easing function."""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in function."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out function."""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out function."""
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in function."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out function."""
    return 1 + (t - 1) * (t - 1) * (t - 1)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out function."""
    return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1


# Dictionary of available easing functions
EASING_FUNCTIONS = {
    'linear': linear_ease,
    'ease_in_quad': ease_in_quad,
    'ease_out_quad': ease_out_quad,
    'ease_in_out_quad': ease_in_out_quad,
    'ease_in_cubic': ease_in_cubic,
    'ease_out_cubic': ease_out_cubic,
    'ease_in_out_cubic': ease_in_out_cubic,
}


def _interpolate_values(from_val: np.ndarray, to_val: np.ndarray, t: float, 
                        easing_func: Callable[[float], float]) -> np.ndarray:
    """
    Interpolate between two arrays of values using an easing function.
    
    Parameters
    ----------
    from_val : array-like
        Starting values
    to_val : array-like
        Ending values
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
        
    Returns
    -------
    array-like
        Interpolated values
    """
    t_eased = easing_func(t)
    return from_val + (to_val - from_val) * t_eased


def _interpolate_color(from_color, to_color, t: float, 
                      easing_func: Callable[[float], float]) -> Union[str, Tuple]:
    """
    Interpolate between two colors using an easing function.
    
    Parameters
    ----------
    from_color : color
        Starting color (can be any matplotlib color format)
    to_color : color
        Ending color (can be any matplotlib color format)
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
        
    Returns
    -------
    color
        Interpolated color
    """
    t_eased = easing_func(t)
    
    # Convert colors to RGBA
    from_rgba = mcolors.to_rgba(from_color)
    to_rgba = mcolors.to_rgba(to_color)
    
    # Interpolate RGBA values
    r = from_rgba[0] + (to_rgba[0] - from_rgba[0]) * t_eased
    g = from_rgba[1] + (to_rgba[1] - from_rgba[1]) * t_eased
    b = from_rgba[2] + (to_rgba[2] - from_rgba[2]) * t_eased
    a = from_rgba[3] + (to_rgba[3] - from_rgba[3]) * t_eased
    
    return (r, g, b, a)


def _interpolate_colors(from_colors, to_colors, t: float, 
                       easing_func: Callable[[float], float]) -> List:
    """
    Interpolate between two lists of colors using an easing function.
    
    Parameters
    ----------
    from_colors : list of colors
        Starting colors
    to_colors : list of colors
        Ending colors
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
        
    Returns
    -------
    list
        Interpolated colors
    """
    if isinstance(from_colors, (list, tuple, np.ndarray)) and isinstance(to_colors, (list, tuple, np.ndarray)):
        return [_interpolate_color(fc, tc, t, easing_func) 
                for fc, tc in zip(from_colors, to_colors)]
    else:
        return _interpolate_color(from_colors, to_colors, t, easing_func)


def _update_line_data(line, from_data, to_data, t: float, 
                     easing_func: Callable[[float], float]) -> None:
    """
    Update a line plot with interpolated data.
    
    Parameters
    ----------
    line : matplotlib.lines.Line2D
        Line object to update
    from_data : tuple of (x, y)
        Starting data
    to_data : tuple of (x, y)
        Ending data
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
    """
    from_x, from_y = from_data
    to_x, to_y = to_data
    
    # Interpolate x and y data
    x = _interpolate_values(from_x, to_x, t, easing_func)
    y = _interpolate_values(from_y, to_y, t, easing_func)
    
    # Update line data
    line.set_data(x, y)


def _update_scatter_data(scatter, from_data, to_data, t: float, 
                        easing_func: Callable[[float], float]) -> None:
    """
    Update a scatter plot with interpolated data.
    
    Parameters
    ----------
    scatter : matplotlib.collections.PathCollection
        Scatter object to update
    from_data : dict
        Starting data with keys 'x', 'y', 'sizes', 'colors'
    to_data : dict
        Ending data with keys 'x', 'y', 'sizes', 'colors'
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
    """
    # Interpolate positions
    x = _interpolate_values(from_data['x'], to_data['x'], t, easing_func)
    y = _interpolate_values(from_data['y'], to_data['y'], t, easing_func)
    
    # Update positions
    scatter.set_offsets(np.column_stack([x, y]))
    
    # Update sizes if provided
    if 'sizes' in from_data and 'sizes' in to_data:
        sizes = _interpolate_values(from_data['sizes'], to_data['sizes'], t, easing_func)
        scatter.set_sizes(sizes)
    
    # Update colors if provided
    if 'colors' in from_data and 'colors' in to_data:
        colors = _interpolate_colors(from_data['colors'], to_data['colors'], t, easing_func)
        scatter.set_facecolor(colors)
        scatter.set_edgecolor(colors)


def _update_bar_data(bars, from_data, to_data, t: float, 
                    easing_func: Callable[[float], float]) -> None:
    """
    Update a bar plot with interpolated data.
    
    Parameters
    ----------
    bars : matplotlib.container.BarContainer
        Bar container to update
    from_data : dict
        Starting data with keys 'heights', 'widths', 'positions', 'colors'
    to_data : dict
        Ending data with keys 'heights', 'widths', 'positions', 'colors'
    t : float
        Interpolation parameter (0 to 1)
    easing_func : callable
        Easing function to use for interpolation
    """
    # Interpolate heights
    heights = _interpolate_values(from_data['heights'], to_data['heights'], t, easing_func)
    
    # Update heights
    for bar, height in zip(bars, heights):
        bar.set_height(height)
    
    # Update widths if provided
    if 'widths' in from_data and 'widths' in to_data:
        widths = _interpolate_values(from_data['widths'], to_data['widths'], t, easing_func)
        for bar, width in zip(bars, widths):
            bar.set_width(width)
    
    # Update positions if provided
    if 'positions' in from_data and 'positions' in to_data:
        positions = _interpolate_values(from_data['positions'], to_data['positions'], t, easing_func)
        for bar, pos in zip(bars, positions):
            bar.set_x(pos)
    
    # Update colors if provided
    if 'colors' in from_data and 'colors' in to_data:
        colors = _interpolate_colors(from_data['colors'], to_data['colors'], t, easing_func)
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
            bar.set_edgecolor(color)


def smooth_transition(from_data: Dict, to_data: Dict, 
                     duration: float = 1.0, fps: int = 30, 
                     easing: str = 'ease_in_out_cubic',
                     plot_type: str = 'line',
                     fig: Optional[Figure] = None, 
                     ax: Optional[plt.Axes] = None,
                     **kwargs) -> animation.FuncAnimation:
    """
    Create a smooth animation transitioning between two data states.
    
    Parameters
    ----------
    from_data : dict
        Starting data state. Format depends on plot_type:
        - 'line': {'x': array, 'y': array, 'color': color, 'linewidth': float}
        - 'scatter': {'x': array, 'y': array, 'sizes': array, 'colors': array or color}
        - 'bar': {'heights': array, 'widths': array, 'positions': array, 'colors': array or color}
    to_data : dict
        Ending data state (same format as from_data)
    duration : float, default: 1.0
        Duration of the transition in seconds
    fps : int, default: 30
        Frames per second
    easing : str, default: 'ease_in_out_cubic'
        Easing function to use. Options: 'linear', 'ease_in_quad', 'ease_out_quad',
        'ease_in_out_quad', 'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic'
    plot_type : str, default: 'line'
        Type of plot to create. Options: 'line', 'scatter', 'bar'
    fig : matplotlib.figure.Figure, optional
        Figure to use. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to use. If None, a new axes is created.
    **kwargs
        Additional keyword arguments to pass to the plot function
        
    Returns
    -------
    animation.FuncAnimation
        Animation object
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.transitions import smooth_transition
    >>> 
    >>> # Line plot transition
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> from_data = {'x': x, 'y': np.sin(x)}
    >>> to_data = {'x': x, 'y': np.cos(x)}
    >>> ani = smooth_transition(from_data, to_data, plot_type='line')
    >>> plt.show()
    """
    if easing not in EASING_FUNCTIONS:
        raise ValueError(f"Unknown easing function: {easing}. "
                         f"Available options: {list(EASING_FUNCTIONS.keys())}")
    
    easing_func = EASING_FUNCTIONS[easing]
    
    # Create figure and axes if not provided
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    
    # Create initial plot based on plot_type
    if plot_type == 'line':
        line, = ax.plot(from_data['x'], from_data['y'], **kwargs)
        
        # Set up the update function for line plot
        def update(frame):
            t = frame / (duration * fps)
            _update_line_data(line, (from_data['x'], from_data['y']), 
                             (to_data['x'], to_data['y']), t, easing_func)
            
            # Update color if provided
            if 'color' in from_data and 'color' in to_data:
                color = _interpolate_color(from_data['color'], to_data['color'], t, easing_func)
                line.set_color(color)
            
            # Update linewidth if provided
            if 'linewidth' in from_data and 'linewidth' in to_data:
                lw = _interpolate_values(np.array([from_data['linewidth']]), 
                                        np.array([to_data['linewidth']]), t, easing_func)[0]
                line.set_linewidth(lw)
            
            return [line]
    
    elif plot_type == 'scatter':
        scatter = ax.scatter(from_data['x'], from_data['y'], 
                           s=from_data.get('sizes', None),
                           c=from_data.get('colors', None),
                           **kwargs)
        
        # Set up the update function for scatter plot
        def update(frame):
            t = frame / (duration * fps)
            _update_scatter_data(scatter, from_data, to_data, t, easing_func)
            return [scatter]
    
    elif plot_type == 'bar':
        bars = ax.bar(from_data.get('positions', range(len(from_data['heights']))),
                     from_data['heights'],
                     width=from_data.get('widths', 0.8),
                     color=from_data.get('colors', None),
                     **kwargs)
        
        # Set up the update function for bar plot
        def update(frame):
            t = frame / (duration * fps)
            _update_bar_data(bars, from_data, to_data, t, easing_func)
            return bars
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. "
                         f"Available options: 'line', 'scatter', 'bar'")
    
    # Create animation
    frames = int(duration * fps)
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                 interval=1000/fps, blit=True)
    
    return ani


def transition_plot_state(fig_from: Figure, fig_to: Figure, 
                         duration: float = 1.0, fps: int = 30,
                         easing: str = 'ease_in_out_cubic') -> animation.FuncAnimation:
    """
    Create a smooth animation transitioning between two figure states.
    
    Parameters
    ----------
    fig_from : matplotlib.figure.Figure
        Starting figure state
    fig_to : matplotlib.figure.Figure
        Ending figure state
    duration : float, default: 1.0
        Duration of the transition in seconds
    fps : int, default: 30
        Frames per second
    easing : str, default: 'ease_in_out_cubic'
        Easing function to use. Options: 'linear', 'ease_in_quad', 'ease_out_quad',
        'ease_in_out_quad', 'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic'
        
    Returns
    -------
    animation.FuncAnimation
        Animation object
    
    Notes
    -----
    This function creates a new figure that transitions between the two input figures.
    The input figures should have the same structure (same number of axes, same types
    of plots in each axes) for the transition to work correctly.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.transitions import transition_plot_state
    >>> 
    >>> # Create first figure
    >>> fig1 = plt.figure()
    >>> ax1 = fig1.add_subplot(111)
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> ax1.plot(x, np.sin(x))
    >>> 
    >>> # Create second figure
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.plot(x, np.cos(x))
    >>> 
    >>> # Create transition
    >>> ani = transition_plot_state(fig1, fig2)
    >>> plt.show()
    """
    if easing not in EASING_FUNCTIONS:
        raise ValueError(f"Unknown easing function: {easing}. "
                         f"Available options: {list(EASING_FUNCTIONS.keys())}")
    
    easing_func = EASING_FUNCTIONS[easing]
    
    # Create a new figure for the animation
    fig = plt.figure(figsize=fig_from.get_size_inches())
    
    # Get all axes from both figures
    axes_from = fig_from.get_axes()
    axes_to = fig_to.get_axes()
    
    if len(axes_from) != len(axes_to):
        raise ValueError("The two figures must have the same number of axes")
    
    # Create corresponding axes in the new figure
    axes = []
    for ax_from in axes_from:
        # Get the position of the original axes
        pos = ax_from.get_position()
        # Create a new axes with the same position
        ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
        axes.append(ax)
    
    # Create a list to store all artists that need to be updated
    artists = []
    
    # For each pair of axes, find corresponding artists and set up transitions
    for ax, ax_from, ax_to in zip(axes, axes_from, axes_to):
        # Copy the initial state from the first figure
        for line in ax_from.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            color = line.get_color()
            linewidth = line.get_linewidth()
            
            # Create a new line in the animation figure
            new_line, = ax.plot(x_data, y_data, color=color, linewidth=linewidth)
            artists.append(new_line)
        
        # Store scatter plots
        for sc in ax_from.collections:
            if isinstance(sc, plt.matplotlib.collections.PathCollection):  # Scatter plot
                offsets = sc.get_offsets()
                sizes = sc.get_sizes()
                colors = sc.get_facecolor()
                
                # Create a new scatter in the animation figure
                new_sc = ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=colors)
                artists.append(new_sc)
        
        # Store bar plots
        for container in ax_from.containers:
            if isinstance(container, plt.matplotlib.container.BarContainer):
                heights = [bar.get_height() for bar in container]
                widths = [bar.get_width() for bar in container]
                positions = [bar.get_x() for bar in container]
                colors = [bar.get_facecolor() for bar in container]
                
                # Create new bars in the animation figure
                new_bars = ax.bar(positions, heights, width=widths, color=colors)
                artists.extend(new_bars)
    
    # Set up the update function
    def update(frame):
        t = frame / (duration * fps)
        t_eased = easing_func(t)
        
        updated_artists = []
        
        # Update each pair of axes
        for ax, ax_from, ax_to in zip(axes, axes_from, axes_to):
            # Update lines
            for i, (line_from, line_to) in enumerate(zip(ax_from.get_lines(), ax_to.get_lines())):
                line = ax.get_lines()[i]
                
                # Interpolate data
                x_from = line_from.get_xdata()
                y_from = line_from.get_ydata()
                x_to = line_to.get_xdata()
                y_to = line_to.get_ydata()
                
                x = _interpolate_values(x_from, x_to, t, easing_func)
                y = _interpolate_values(y_from, y_to, t, easing_func)
                
                # Update line data
                line.set_data(x, y)
                
                # Interpolate color
                color_from = mcolors.to_rgba(line_from.get_color())
                color_to = mcolors.to_rgba(line_to.get_color())
                color = _interpolate_color(color_from, color_to, t, easing_func)
                line.set_color(color)
                
                # Interpolate linewidth
                lw_from = line_from.get_linewidth()
                lw_to = line_to.get_linewidth()
                lw = lw_from + (lw_to - lw_from) * t_eased
                line.set_linewidth(lw)
                
                updated_artists.append(line)
            
            # Update scatter plots
            sc_from_list = [sc for sc in ax_from.collections 
                          if isinstance(sc, plt.collections.PathCollection)]
            sc_to_list = [sc for sc in ax_to.collections 
                        if isinstance(sc, plt.collections.PathCollection)]
            
            for i, (sc_from, sc_to) in enumerate(zip(sc_from_list, sc_to_list)):
                sc_idx = len(ax.get_lines()) + i
                sc = ax.collections[sc_idx - len(ax.get_lines())]
                
                # Interpolate offsets
                offsets_from = sc_from.get_offsets()
                offsets_to = sc_to.get_offsets()
                
                # Handle different array shapes
                if offsets_from.shape != offsets_to.shape:
                    # Use the smaller shape and pad the larger one
                    min_shape = min(offsets_from.shape[0], offsets_to.shape[0])
                    offsets_from = offsets_from[:min_shape]
                    offsets_to = offsets_to[:min_shape]
                
                offsets = _interpolate_values(offsets_from, offsets_to, t, easing_func)
                sc.set_offsets(offsets)
                
                # Interpolate sizes
                sizes_from = sc_from.get_sizes()
                sizes_to = sc_to.get_sizes()
                
                # Handle different array shapes
                if sizes_from.shape != sizes_to.shape:
                    min_shape = min(sizes_from.shape[0], sizes_to.shape[0])
                    sizes_from = sizes_from[:min_shape]
                    sizes_to = sizes_to[:min_shape]
                
                sizes = _interpolate_values(sizes_from, sizes_to, t, easing_func)
                sc.set_sizes(sizes)
                
                # Interpolate colors
                colors_from = sc_from.get_facecolor()
                colors_to = sc_to.get_facecolor()
                
                # Handle different array shapes
                if colors_from.shape != colors_to.shape:
                    min_shape = min(colors_from.shape[0], colors_to.shape[0])
                    colors_from = colors_from[:min_shape]
                    colors_to = colors_to[:min_shape]
                
                colors = np.zeros_like(colors_from)
                for j in range(colors_from.shape[0]):
                    colors[j] = _interpolate_color(colors_from[j], colors_to[j], t, easing_func)
                
                sc.set_facecolor(colors)
                sc.set_edgecolor(colors)
                
                updated_artists.append(sc)
            
            # Update bar plots
            container_from_list = [c for c in ax_from.containers 
                                 if isinstance(c, plt.container.BarContainer)]
            container_to_list = [c for c in ax_to.containers 
                               if isinstance(c, plt.container.BarContainer)]
            
            for i, (container_from, container_to) in enumerate(zip(container_from_list, container_to_list)):
                # Get the corresponding bars in the animation figure
                start_idx = len(ax.get_lines()) + len([sc for sc in ax.collections 
                                                    if isinstance(sc, plt.matplotlib.collections.PathCollection)])
                for j, (bar_from, bar_to) in enumerate(zip(container_from, container_to)):
                    bar_idx = start_idx + j
                    if bar_idx < len(ax.patches):
                        bar = ax.patches[bar_idx]
                        
                        # Interpolate height
                        height_from = bar_from.get_height()
                        height_to = bar_to.get_height()
                        height = height_from + (height_to - height_from) * t_eased
                        bar.set_height(height)
                        
                        # Interpolate width
                        width_from = bar_from.get_width()
                        width_to = bar_to.get_width()
                        width = width_from + (width_to - width_from) * t_eased
                        bar.set_width(width)
                        
                        # Interpolate position
                        x_from = bar_from.get_x()
                        x_to = bar_to.get_x()
                        x = x_from + (x_to - x_from) * t_eased
                        bar.set_x(x)
                        
                        # Interpolate color
                        color_from = mcolors.to_rgba(bar_from.get_facecolor()[0])
                        color_to = mcolors.to_rgba(bar_to.get_facecolor()[0])
                        color = _interpolate_color(color_from, color_to, t, easing_func)
                        bar.set_facecolor(color)
                        bar.set_edgecolor(color)
                        
                        updated_artists.append(bar)
        
        return updated_artists
    
    # Create animation
    frames = int(duration * fps)
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                 interval=1000/fps, blit=True)
    
    return ani