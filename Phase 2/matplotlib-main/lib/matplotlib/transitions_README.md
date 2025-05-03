# Matplotlib Smooth Transitions

This module provides functionality to create smooth animations/transitions between different states of a plot in Matplotlib.

## Features

- Smooth transitions between different data states
- Support for various plot types (line, scatter, bar)
- Customizable transition duration and frames per second
- Multiple easing functions (linear, ease-in, ease-out, etc.)
- Transition of various plot properties:
  - Data values (y-values in line plots, heights in bar charts, etc.)
  - Colors
  - Sizes/widths
  - Positions

## Functions

### `smooth_transition(from_data, to_data, duration=1.0, fps=30, **kwargs)`

Creates a smooth animation transitioning between two data states.

#### Parameters

- `from_data` (dict): Starting data state. Format depends on plot_type:
  - 'line': {'x': array, 'y': array, 'color': color, 'linewidth': float}
  - 'scatter': {'x': array, 'y': array, 'sizes': array, 'colors': array or color}
  - 'bar': {'heights': array, 'widths': array, 'positions': array, 'colors': array or color}
- `to_data` (dict): Ending data state (same format as from_data)
- `duration` (float, default: 1.0): Duration of the transition in seconds
- `fps` (int, default: 30): Frames per second
- `easing` (str, default: 'ease_in_out_cubic'): Easing function to use. Options: 'linear', 'ease_in_quad', 'ease_out_quad', 'ease_in_out_quad', 'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic'
- `plot_type` (str, default: 'line'): Type of plot to create. Options: 'line', 'scatter', 'bar'
- `fig` (matplotlib.figure.Figure, optional): Figure to use. If None, a new figure is created.
- `ax` (matplotlib.axes.Axes, optional): Axes to use. If None, a new axes is created.
- `**kwargs`: Additional keyword arguments to pass to the plot function

#### Returns

- `animation.FuncAnimation`: Animation object

### `transition_plot_state(fig_from, fig_to, duration=1.0, fps=30)`

Creates a smooth animation transitioning between two completely different figure states.

#### Parameters

- `fig_from` (matplotlib.figure.Figure): Starting figure state
- `fig_to` (matplotlib.figure.Figure): Ending figure state
- `duration` (float, default: 1.0): Duration of the transition in seconds
- `fps` (int, default: 30): Frames per second
- `easing` (str, default: 'ease_in_out_cubic'): Easing function to use

#### Returns

- `animation.FuncAnimation`: Animation object

## Examples

### Line Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Create data
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

# Create transition
ani = smooth_transition(
    from_data, to_data,
    duration=2.0, fps=30,
    easing='ease_in_out_cubic',
    plot_type='line'
)

plt.show()
```

### Scatter Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Generate random data
np.random.seed(42)
n_points = 50
from_x = np.random.normal(0, 1, n_points)
from_y = np.random.normal(0, 1, n_points)
from_sizes = np.random.uniform(10, 50, n_points)
from_colors = np.random.uniform(0, 1, (n_points, 4))

to_x = np.random.normal(0, 2, n_points)
to_y = np.random.normal(0, 2, n_points)
to_sizes = np.random.uniform(30, 100, n_points)
to_colors = np.random.uniform(0, 1, (n_points, 4))

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

# Create transition
ani = smooth_transition(
    from_scatter_data, to_scatter_data,
    duration=2.0, fps=30,
    easing='ease_out_quad',
    plot_type='scatter'
)

plt.show()
```

### Bar Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Create data
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

# Create transition
ani = smooth_transition(
    from_bar_data, to_bar_data,
    duration=2.0, fps=30,
    easing='ease_in_out_cubic',
    plot_type='bar'
)

plt.show()
```

### Complete Figure Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import transition_plot_state

# Create first figure
fig_from = plt.figure()
ax1_from = fig_from.add_subplot(121)
x = np.linspace(0, 2*np.pi, 100)
ax1_from.plot(x, np.sin(x), 'b-', linewidth=2)

ax2_from = fig_from.add_subplot(122)
ax2_from.bar(np.arange(5), [3, 5, 2, 7, 4], width=0.6, color='skyblue')

# Create second figure
fig_to = plt.figure()
ax1_to = fig_to.add_subplot(121)
ax1_to.plot(x, np.cos(x), 'r-', linewidth=3)

ax2_to = fig_to.add_subplot(122)
ax2_to.bar(np.arange(5) + 0.2, [6, 2, 8, 3, 5], width=0.8, color='salmon')

# Create transition
ani = transition_plot_state(fig_from, fig_to, duration=2.0, fps=30)

plt.show()
```

## Saving Animations

You can save the animations using the standard Matplotlib animation saving methods:

```python
# Save as MP4
ani.save('animation.mp4', writer='ffmpeg', fps=30, dpi=100)

# Save as GIF
ani.save('animation.gif', writer='pillow', fps=30, dpi=100)

# Save as HTML
html = ani.to_jshtml()
with open('animation.html', 'w') as f:
    f.write(html)
```