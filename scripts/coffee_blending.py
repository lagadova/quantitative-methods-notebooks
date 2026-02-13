"""
Coffee Blending Optimization - Interactive Sensitivity Analysis Module

This module provides functions for solving a coffee blending linear programming
problem and visualizing the solution with interactive Plotly plots and sliders.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from ipywidgets import FloatSlider, VBox, Label, Output
from IPython.display import display, clear_output


def solve_coffee_problem(arabika_avail, robusta_avail, delux_demand):
    """
    Solve the coffee blending problem with given constraint RHS values.

    Constraints:
    1. 0.5*x1 + 0.25*x2 <= arabika_avail
    2. 0.5*x1 + 0.75*x2 <= robusta_avail
    3. x2 <= delux_demand
    4. x1, x2 >= 0

    Args:
        arabika_avail: Available Arabika coffee (kg)
        robusta_avail: Available Robusta coffee (kg)
        delux_demand: Demand for Delux coffee (kg)

    Returns:
        tuple: (x1_opt, x2_opt, profit, feasible_status)
    """
    # Linear programming formulation (minimization)
    c = [-40, -50]  # Negative because linprog minimizes

    A_ub = [
        [0.5, 0.25],   # Arabika constraint
        [0.5, 0.75],   # Robusta constraint
        [0, 1]         # Delux demand constraint
    ]

    b_ub = [arabika_avail, robusta_avail, delux_demand]

    bounds = [(0, None), (0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x1, x2 = result.x
        profit = -result.fun
        return x1, x2, profit, True
    else:
        return None, None, None, False


def get_feasible_region(arabika_avail, robusta_avail, delux_demand, x_max=300, y_max=200):
    """
    Generate points for plotting the feasible region.

    Args:
        arabika_avail: Available Arabika coffee (kg)
        robusta_avail: Available Robusta coffee (kg)
        delux_demand: Demand for Delux coffee (kg)
        x_max: Maximum x1 value for plotting
        y_max: Maximum x2 value for plotting

    Returns:
        tuple: (x_range, y1, y2, y3, y4) constraint line values
    """
    x_range = np.linspace(0, x_max, 1000)

    # Constraint 1: 0.5*x1 + 0.25*x2 <= arabika_avail
    y1 = (arabika_avail - 0.5*x_range) / 0.25

    # Constraint 2: 0.5*x1 + 0.75*x2 <= robusta_avail
    y2 = (robusta_avail - 0.5*x_range) / 0.75

    # Constraint 3: x2 <= delux_demand
    y3 = np.full_like(x_range, delux_demand)

    # Constraint 4: x2 >= 0
    y4 = np.zeros_like(x_range)

    return x_range, y1, y2, y3, y4


def create_interactive_plot(arabika_avail=120, robusta_avail=160, delux_demand=150):
    """
    Create an interactive Plotly figure for the coffee blending problem.

    Args:
        arabika_avail: Available Arabika coffee (kg)
        robusta_avail: Available Robusta coffee (kg)
        delux_demand: Demand for Delux coffee (kg)

    Returns:
        go.Figure: Plotly figure object
    """

    x_range, y1, y2, y3, y4 = get_feasible_region(
        arabika_avail, robusta_avail, delux_demand)

    # Solve the problem
    x1_opt, x2_opt, profit, feasible = solve_coffee_problem(
        arabika_avail, robusta_avail, delux_demand)

    # Create figure
    fig = go.Figure()

    # Add constraint lines
    fig.add_trace(go.Scatter(
        x=x_range, y=y1,
        name='Arabika: 0.5x₁ + 0.25x₂ ≤ RHS',
        line=dict(color='red', dash='dash'),
        hovertemplate='<b>Arabika constraint</b><br>x₁=%{x:.1f}<br>x₂=%{y:.1f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_range, y=y2,
        name='Robusta: 0.5x₁ + 0.75x₂ ≤ RHS',
        line=dict(color='blue', dash='dash'),
        hovertemplate='<b>Robusta constraint</b><br>x₁=%{x:.1f}<br>x₂=%{y:.1f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_range, y=y3,
        name='Delux demand: x₂ ≤ RHS',
        line=dict(color='green', dash='dash'),
        hovertemplate='<b>Delux demand</b><br>x₁=%{x:.1f}<br>x₂=%{y:.1f}<extra></extra>'
    ))

    # Add feasible region (simplified - using corner points)
    # Find corner points
    corners = []

    # Origin
    corners.append((0, 0))

    # Intersection with x-axis (constraint 1): 0.5*x = arabika_avail => x = 2*arabika_avail
    x_max_arabika = 2 * arabika_avail
    corners.append((min(x_max_arabika, 2*robusta_avail), 0))

    # Intersection with y-axis (constraint 2): 0.75*x2 = robusta_avail => x2 = robusta_avail/0.75
    y_max_robusta = robusta_avail / 0.75
    corners.append((0, min(y_max_robusta, delux_demand)))

    # Intersection of constraints 1 and 2
    # 0.5*x1 + 0.25*x2 = arabika_avail
    # 0.5*x1 + 0.75*x2 = robusta_avail
    # Subtracting: -0.5*x2 = arabika_avail - robusta_avail
    x2_intersect = (robusta_avail - arabika_avail) / 0.5
    x1_intersect = 2 * (arabika_avail - 0.25*x2_intersect)
    if x1_intersect >= 0 and x2_intersect >= 0 and x2_intersect <= delux_demand:
        corners.append((x1_intersect, x2_intersect))

    # Intersection of constraints 1 and 3
    # 0.5*x1 + 0.25*x2 = arabika_avail with x2 = delux_demand
    x1_delux = 2 * (arabika_avail - 0.25*delux_demand)
    if x1_delux >= 0:
        corners.append((x1_delux, delux_demand))

    # Intersection of constraints 2 and 3
    # 0.5*x1 + 0.75*x2 = robusta_avail with x2 = delux_demand
    x1_robusta_delux = 2 * (robusta_avail - 0.75*delux_demand)
    if x1_robusta_delux >= 0:
        corners.append((x1_robusta_delux, delux_demand))

    # Helper function to check if a point is feasible
    def is_feasible_point(x, y, tol=0.01):
        return (x >= -tol and y >= -tol and
                0.5*x + 0.25*y <= arabika_avail + tol and
                0.5*x + 0.75*y <= robusta_avail + tol and
                y <= delux_demand + tol)

    # Filter valid corners - only keep points that satisfy ALL constraints
    valid_corners = [c for c in corners if is_feasible_point(c[0], c[1])]

    if valid_corners:
        # Sort corners by angle to create proper polygon
        center_x = np.mean([c[0] for c in valid_corners])
        center_y = np.mean([c[1] for c in valid_corners])
        valid_corners = sorted(valid_corners,
                               key=lambda p: np.arctan2(p[1]-center_y, p[0]-center_x))

        # Close the polygon
        valid_corners.append(valid_corners[0])

        poly_x = [c[0] for c in valid_corners]
        poly_y = [c[1] for c in valid_corners]

        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            fill='toself',
            name='Feasible Region',
            fillcolor='rgba(0, 100, 200, 0.3)',
            line=dict(color='black', width=2),
            hovertemplate='<b>Feasible region boundary</b><br>x₁=%{x:.1f}<br>x₂=%{y:.1f}<extra></extra>'
        ))

    # Add optimal point if feasible
    if feasible and x1_opt is not None:
        fig.add_trace(go.Scatter(
            x=[x1_opt], y=[x2_opt],
            mode='markers',
            name=f'Optimal: ({x1_opt:.1f}, {x2_opt:.1f})<br>Profit: ${profit:.0f}',
            marker=dict(size=15, color='gold', symbol='star',
                        line=dict(width=2, color='black')),
            hovertemplate='<b>Optimal Solution</b><br>x₁=%{x:.1f}<br>x₂=%{y:.1f}<br>Profit: $' +
            f'{profit:.0f}' + '<extra></extra>'
        ))

    # Add objective function iso-profit lines
    x_profit = np.linspace(0, 400, 100)
    for profit_val in [2000, 4000, 6000, 8000, 10000, 12000]:
        y_profit = (profit_val - 40*x_profit) / 50
        fig.add_trace(go.Scatter(
            x=x_profit, y=y_profit,
            name=f'Profit = ${profit_val}',
            line=dict(color='gray', width=1, dash='dot'),
            hovertemplate='<b>Iso-profit line</b><br>Profit: $' +
            f'{profit_val}' + '<extra></extra>',
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title='Coffee Blending Problem - Interactive Sensitivity Analysis<br><sub>Adjust the constraints below</sub>',
        xaxis_title='x₁ (Standard Coffee in kg)',
        yaxis_title='x₂ (Delux Coffee in kg)',
        hovermode='closest',
        width=900,
        height=700,
        xaxis=dict(range=[0, 400]),
        yaxis=dict(range=[0, 200]),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_interactive_sliders():
    """
    Create interactive sliders for sensitivity analysis.

    Returns:
        None: Displays the interactive widget with sliders and plot.
    """
    # Create sliders
    arabika_slider = FloatSlider(
        value=120,
        min=50,
        max=300,
        step=5,
        description='Arabika (kg):',
        continuous_update=True
    )

    robusta_slider = FloatSlider(
        value=160,
        min=50,
        max=300,
        step=5,
        description='Robusta (kg):',
        continuous_update=True
    )

    delux_slider = FloatSlider(
        value=150,
        min=50,
        max=300,
        step=5,
        description='Delux Demand (kg):',
        continuous_update=True
    )

    # Create initial figure
    output = Output()

    # Update function
    def update_plot(change=None):
        arabika = arabika_slider.value
        robusta = robusta_slider.value
        delux = delux_slider.value

        # Create new figure
        new_fig = create_interactive_plot(arabika, robusta, delux)

        # Clear and display the new figure
        with output:
            clear_output(wait=True)
            display(new_fig)

    # Attach observes to sliders
    arabika_slider.observe(update_plot, names='value')
    robusta_slider.observe(update_plot, names='value')
    delux_slider.observe(update_plot, names='value')

    # Initial plot
    update_plot()

    # Display sliders and plot
    display(VBox([
        Label('Adjust the constraint right-hand sides (RHS):'),
        arabika_slider,
        robusta_slider,
        delux_slider,
        Label('Interactive Plot:'),
        output
    ]))
