"""
======================
Smooth Transitions Demo
======================

This example demonstrates the use of the new smooth transition functionality
in matplotlib, which allows for creating smooth animations between different
states of a plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the matplotlib-main directory to the Python path
sys.path.insert(0, os.path.abspath('/workspace/user-tasks-3/Phase 2/matplotlib-main'))

# Import our custom transitions module
from lib.matplotlib.transitions import smooth_transition, transition_plot_state
import matplotlib.animation as animation
from matplotlib.colors import to_rgba

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
import http.server
import socketserver
import threading
import webbrowser
import os

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