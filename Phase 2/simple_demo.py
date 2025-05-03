#!/usr/bin/env python3
"""
Simple demo for the smooth transitions functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import the transitions module directly
sys.path.insert(0, os.path.dirname(__file__))
from matplotlib_main_lib_matplotlib_transitions import smooth_transition, transition_plot_state


def demo_line_transition():
    """Demo of a line plot transition."""
    print("Demo 1: Line Plot Transition")
    
    # Create data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create from and to data dictionaries
    from_data = {'x': x, 'y': y1, 'color': 'blue', 'linewidth': 1.0}
    to_data = {'x': x, 'y': y2, 'color': 'red', 'linewidth': 3.0}
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Line Plot Transition")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    
    ani = smooth_transition(from_data, to_data, duration=2.0, fig=fig, ax=ax)
    
    plt.tight_layout()
    plt.show()


def demo_scatter_transition():
    """Demo of a scatter plot transition."""
    print("Demo 2: Scatter Plot Transition")
    
    # Create data
    np.random.seed(42)  # For reproducibility
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Scatter Plot Transition")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    
    ani = smooth_transition(from_data, to_data, duration=2.0, 
                           plot_type='scatter', fig=fig, ax=ax)
    
    plt.tight_layout()
    plt.show()


def demo_bar_transition():
    """Demo of a bar plot transition."""
    print("Demo 3: Bar Plot Transition")
    
    # Create data
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [3, 7, 2, 5, 8]
    values2 = [8, 4, 6, 2, 5]
    
    # Create from and to data dictionaries
    from_data = {'heights': values1, 'colors': 'skyblue'}
    to_data = {'heights': values2, 'colors': 'salmon'}
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Bar Plot Transition")
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    
    ani = smooth_transition(from_data, to_data, duration=2.0, 
                           plot_type='bar', fig=fig, ax=ax)
    
    plt.tight_layout()
    plt.show()


def demo_easing_functions():
    """Demo of different easing functions."""
    print("Demo 4: Different Easing Functions")
    
    # Create data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create from and to data dictionaries
    from_data = {'x': x, 'y': y1}
    to_data = {'x': x, 'y': y2}
    
    # Create animations with different easing functions
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    
    easing_functions = [
        'linear', 'ease_in_quad', 'ease_out_quad',
        'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic'
    ]
    
    animations = []
    for i, easing in enumerate(easing_functions):
        ax = axes[i]
        ax.set_title(easing)
        ax.grid(True)
        
        ani = smooth_transition(from_data, to_data, duration=2.0, 
                               easing=easing, fig=fig, ax=ax)
        animations.append(ani)
    
    plt.tight_layout()
    plt.show()


def demo_figure_transition():
    """Demo of a complete figure transition."""
    print("Demo 5: Complete Figure Transition")
    
    # Create first figure
    fig1 = plt.figure(figsize=(8, 4))
    ax1 = fig1.add_subplot(111)
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'b-', linewidth=2)
    ax1.set_title("Figure 1: Sine Wave")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)
    
    # Create second figure
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, np.cos(x), 'r-', linewidth=3)
    ax2.set_title("Figure 2: Cosine Wave")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    
    # Create transition animation
    ani = transition_plot_state(fig1, fig2, duration=2.0)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demos."""
    print("Matplotlib Smooth Transitions Demo")
    print("==================================")
    
    # Run demos
    demo_line_transition()
    demo_scatter_transition()
    demo_bar_transition()
    demo_easing_functions()
    demo_figure_transition()


if __name__ == "__main__":
    main()