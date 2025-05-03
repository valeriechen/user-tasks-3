# Matplotlib Smooth Transitions

This project implements smooth transitions/animations between different states of plots in Matplotlib.

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

## Implementation

The implementation consists of two main functions:

1. `smooth_transition(from_data, to_data, duration=1.0, fps=30, **kwargs)`:
   - Takes initial and final data states
   - Creates a smooth animation transitioning between the states
   - Supports different plot types (line, scatter, bar)
   - Allows customization of transition duration and frames per second
   - Supports transitions of various plot properties

2. `transition_plot_state(fig_from, fig_to, duration=1.0, fps=30)`:
   - Takes two complete figure states
   - Creates a smooth animation transitioning between the two figures
   - Supports transitions of all plot elements in the figures

## Usage Examples

### Line Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create from and to data dictionaries
from_data = {'x': x, 'y': y1, 'color': 'blue', 'linewidth': 1.0}
to_data = {'x': x, 'y': y2, 'color': 'red', 'linewidth': 3.0}

# Create animation
fig, ax = plt.subplots()
ani = smooth_transition(from_data, to_data, duration=2.0, fig=fig, ax=ax)

plt.show()
```

### Scatter Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Create data
np.random.seed(42)
x1 = np.random.rand(50)
y1 = np.random.rand(50)
x2 = np.random.rand(50)
y2 = np.random.rand(50)
sizes1 = np.random.rand(50) * 100 + 50
sizes2 = np.random.rand(50) * 200 + 100

# Create from and to data dictionaries
from_data = {'x': x1, 'y': y1, 'sizes': sizes1, 'colors': 'blue'}
to_data = {'x': x2, 'y': y2, 'sizes': sizes2, 'colors': 'red'}

# Create animation
fig, ax = plt.subplots()
ani = smooth_transition(from_data, to_data, duration=2.0, 
                       plot_type='scatter', fig=fig, ax=ax)

plt.show()
```

### Bar Plot Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import smooth_transition

# Create data
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [3, 7, 2, 5, 8]
values2 = [8, 4, 6, 2, 5]

# Create from and to data dictionaries
from_data = {'heights': values1, 'colors': 'skyblue'}
to_data = {'heights': values2, 'colors': 'salmon'}

# Create animation
fig, ax = plt.subplots()
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)
ani = smooth_transition(from_data, to_data, duration=2.0, 
                       plot_type='bar', fig=fig, ax=ax)

plt.show()
```

### Figure Transition

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transitions import transition_plot_state

# Create first figure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'b-', linewidth=2)

# Create second figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, np.cos(x), 'r-', linewidth=3)

# Create transition animation
ani = transition_plot_state(fig1, fig2, duration=2.0)

plt.show()
```

## Testing

The implementation has been tested with various plot types and transition properties. The test script `test_transitions.py` demonstrates the functionality of the implementation.

To run the tests:

```bash
python test_transitions.py
```

## Files

- `matplotlib-main/lib/matplotlib/transitions.py`: Main implementation of the transitions module
- `test_transitions.py`: Test script demonstrating the functionality
- `README.md`: This file

## Future Improvements

- Support for more plot types (pie, contour, etc.)
- Support for more plot properties (markers, linestyles, etc.)
- Support for more complex transitions (morphing between different plot types)
- Support for custom interpolation functions