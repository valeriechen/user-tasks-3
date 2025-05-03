"""
Smooth Transitions Demo (Standalone)

This script demonstrates the smooth transition functionality that would be
implemented in matplotlib. For demonstration purposes, we include the
implementation directly in this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.artist import Artist
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Union, Callable, Optional, Any, Iterable
import copy
import http.server
import socketserver
import threading
import os


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


def _interpolate_values(from_val, to_val, t: float, 
                        easing_func: Callable[[float], float]):
    """
    Interpolate between two arrays of values using an easing function.
    """
    t_eased = easing_func(t)
    
    # Convert lists to numpy arrays if needed
    if isinstance(from_val, (list, tuple)):
        from_val = np.array(from_val)
    if isinstance(to_val, (list, tuple)):
        to_val = np.array(to_val)
        
    return from_val + (to_val - from_val) * t_eased


def _interpolate_color(from_color, to_color, t: float, 
                      easing_func: Callable[[float], float]) -> Union[str, Tuple]:
    """
    Interpolate between two colors using an easing function.
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
            if isinstance(sc, plt.collections.PathCollection):  # Scatter plot
                offsets = sc.get_offsets()
                sizes = sc.get_sizes()
                colors = sc.get_facecolor()
                
                # Create a new scatter in the animation figure
                new_sc = ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=colors)
                artists.append(new_sc)
        
        # Store bar plots
        for container in ax_from.containers:
            if str(type(container)).find('BarContainer') > 0:
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
                                 if str(type(c)).find('BarContainer') > 0]
            container_to_list = [c for c in ax_to.containers 
                               if str(type(c)).find('BarContainer') > 0]
            
            for i, (container_from, container_to) in enumerate(zip(container_from_list, container_to_list)):
                # Get the corresponding bars in the animation figure
                start_idx = len(ax.get_lines()) + len([sc for sc in ax.collections 
                                                    if isinstance(sc, plt.collections.PathCollection)])
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
                        try:
                            color_from = bar_from.get_facecolor()
                            if isinstance(color_from, np.ndarray) and color_from.ndim > 1:
                                color_from = color_from[0]
                            
                            color_to = bar_to.get_facecolor()
                            if isinstance(color_to, np.ndarray) and color_to.ndim > 1:
                                color_to = color_to[0]
                                
                            color = _interpolate_color(color_from, color_to, t, easing_func)
                            bar.set_facecolor(color)
                            bar.set_edgecolor(color)
                        except Exception as e:
                            # Skip color interpolation if there's an error
                            print(f"Warning: Could not interpolate bar colors: {e}")
                            pass
                        
                        updated_artists.append(bar)
        
        return updated_artists
    
    # Create animation
    frames = int(duration * fps)
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                 interval=1000/fps, blit=True)
    
    return ani


# Set up a web server compatible configuration
plt.rcParams['animation.html'] = 'jshtml'

# Demo 1: Line plot transition
print("Demo 1: Line plot transition")
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.set_xlim(0, 2*np.pi)
ax1.set_ylim(-1.5, 1.5)
ax1.set_title('Line Plot Transition')

x = np.linspace(0, 2*np.pi, 100)
from_data = {
    'x': x,
    'y': np.sin(x),
    'color': 'blue',
    'linewidth': 1.0
}
to_data = {
    'x': x,
    'y': np.cos(x),
    'color': 'red',
    'linewidth': 3.0
}

ani1 = smooth_transition(
    from_data, to_data,
    duration=2.0, fps=30,
    easing='ease_in_out_cubic',
    plot_type='line',
    fig=fig1, ax=ax1
)

# Demo 2: Scatter plot transition
print("Demo 2: Scatter plot transition")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_title('Scatter Plot Transition')

# Generate random data
np.random.seed(42)
n_points = 50
from_x = np.random.normal(0, 1, n_points)
from_y = np.random.normal(0, 1, n_points)
from_sizes = np.random.uniform(10, 50, n_points)
from_colors = np.random.uniform(0, 1, (n_points, 4))
from_colors[:, 3] = 0.7  # Alpha

to_x = np.random.normal(0, 2, n_points)
to_y = np.random.normal(0, 2, n_points)
to_sizes = np.random.uniform(30, 100, n_points)
to_colors = np.random.uniform(0, 1, (n_points, 4))
to_colors[:, 3] = 0.9  # Alpha

from_scatter_data = {
    'x': from_x,
    'y': from_y,
    'sizes': from_sizes,
    'colors': from_colors
}
to_scatter_data = {
    'x': to_x,
    'y': to_y,
    'sizes': to_sizes,
    'colors': to_colors
}

ani2 = smooth_transition(
    from_scatter_data, to_scatter_data,
    duration=2.0, fps=30,
    easing='ease_out_quad',
    plot_type='scatter',
    fig=fig2, ax=ax2
)

# Demo 3: Bar plot transition
print("Demo 3: Bar plot transition")
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.set_ylim(0, 12)
ax3.set_title('Bar Plot Transition')

categories = ['A', 'B', 'C', 'D', 'E']
from_heights = [5, 7, 3, 8, 4]
to_heights = [2, 10, 6, 4, 9]

from_positions = np.arange(len(categories))
to_positions = np.arange(len(categories)) + 0.2

from_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
to_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']

from_bar_data = {
    'heights': from_heights,
    'positions': from_positions,
    'widths': np.full(len(categories), 0.5),
    'colors': from_colors
}
to_bar_data = {
    'heights': to_heights,
    'positions': to_positions,
    'widths': np.full(len(categories), 0.7),
    'colors': to_colors
}

ani3 = smooth_transition(
    from_bar_data, to_bar_data,
    duration=2.0, fps=30,
    easing='ease_in_out_cubic',
    plot_type='bar',
    fig=fig3, ax=ax3
)

# Demo 4: Comparing different easing functions
print("Demo 4: Comparing different easing functions")
fig4, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
fig4.suptitle('Different Easing Functions')

easing_functions = [
    'linear',
    'ease_in_quad',
    'ease_out_quad',
    'ease_in_out_quad',
    'ease_in_cubic',
    'ease_out_cubic'
]

for i, easing in enumerate(easing_functions):
    ax = axes[i]
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(easing)
    
    from_data = {
        'x': x,
        'y': np.sin(x),
        'color': 'blue',
        'linewidth': 1.0
    }
    to_data = {
        'x': x,
        'y': np.cos(x),
        'color': 'red',
        'linewidth': 3.0
    }
    
    smooth_transition(
        from_data, to_data,
        duration=2.0, fps=30,
        easing=easing,
        plot_type='line',
        fig=fig4, ax=ax
    )

plt.tight_layout()

# Demo 5: Complete figure transition
print("Demo 5: Complete figure transition")

# Create first figure
fig_from = plt.figure(figsize=(10, 6))
ax1_from = fig_from.add_subplot(121)
ax1_from.set_xlim(0, 2*np.pi)
ax1_from.set_ylim(-1.5, 1.5)
ax1_from.set_title('Sine Wave')
ax1_from.plot(x, np.sin(x), 'b-', linewidth=2)

ax2_from = fig_from.add_subplot(122)
ax2_from.set_xlim(0, 10)
ax2_from.set_ylim(0, 10)
ax2_from.set_title('Bar Chart')
ax2_from.bar(np.arange(5), [3, 5, 2, 7, 4], width=0.6, color='skyblue')

# Create second figure
fig_to = plt.figure(figsize=(10, 6))
ax1_to = fig_to.add_subplot(121)
ax1_to.set_xlim(0, 2*np.pi)
ax1_to.set_ylim(-1.5, 1.5)
ax1_to.set_title('Cosine Wave')
ax1_to.plot(x, np.cos(x), 'r-', linewidth=3)

ax2_to = fig_to.add_subplot(122)
ax2_to.set_xlim(0, 10)
ax2_to.set_ylim(0, 10)
ax2_to.set_title('Updated Bar Chart')
ax2_to.bar(np.arange(5) + 0.2, [6, 2, 8, 3, 5], width=0.8, color='salmon')

# Create transition
ani5 = transition_plot_state(fig_from, fig_to, duration=2.0, fps=30)

# Save animations to HTML for viewing
print("Saving animations to HTML files...")

# Function to save animation to HTML
def save_animation_to_html(ani, filename):
    html = ani.to_jshtml()
    with open(filename, 'w') as f:
        f.write(html)
    print(f"Saved {filename}")

# Save each animation
save_animation_to_html(ani1, 'line_transition.html')
save_animation_to_html(ani2, 'scatter_transition.html')
save_animation_to_html(ani3, 'bar_transition.html')
save_animation_to_html(ani5, 'figure_transition.html')

# Create a combined HTML file with all demos
with open('smooth_transitions_demo.html', 'w') as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Matplotlib Smooth Transitions Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333366; }
            h2 { color: #666699; margin-top: 30px; }
            .demo { margin: 20px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Matplotlib Smooth Transitions Demo</h1>
        
        <h2>Demo 1: Line Plot Transition</h2>
        <div class="demo">
    """)
    
    with open('line_transition.html', 'r') as demo_file:
        f.write(demo_file.read())
    
    f.write("""
        </div>
        
        <h2>Demo 2: Scatter Plot Transition</h2>
        <div class="demo">
    """)
    
    with open('scatter_transition.html', 'r') as demo_file:
        f.write(demo_file.read())
    
    f.write("""
        </div>
        
        <h2>Demo 3: Bar Plot Transition</h2>
        <div class="demo">
    """)
    
    with open('bar_transition.html', 'r') as demo_file:
        f.write(demo_file.read())
    
    f.write("""
        </div>
        
        <h2>Demo 5: Complete Figure Transition</h2>
        <div class="demo">
    """)
    
    with open('figure_transition.html', 'r') as demo_file:
        f.write(demo_file.read())
    
    f.write("""
        </div>
    </body>
    </html>
    """)

print("Created combined demo HTML: smooth_transitions_demo.html")

# Start a simple HTTP server to view the demos
PORT = 12000  # Use the assigned port
DIRECTORY = os.getcwd()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def start_server():
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

# Start the server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

print(f"Server started at http://localhost:{PORT}")
print(f"View the demo at http://localhost:{PORT}/smooth_transitions_demo.html")

# Keep the main thread running
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Server stopped.")