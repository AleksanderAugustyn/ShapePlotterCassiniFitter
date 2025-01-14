"""
Nuclear Shape Plotter using Cassini Ovals - A program to visualize and analyze nuclear shapes.
This version implements an object-oriented design for better organization and maintainability.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy import integrate
from scipy.special import sph_harm

matplotlib.use('TkAgg')


def double_factorial(n):
    """Calculate double factorial n!!"""
    return math.prod(range(n, 0, -2))


def calculate_radius_vector(rho: np.ndarray, z: np.ndarray) -> tuple[Any, Any]:
    """Calculate the radius vector for given shape in cylindrical coordinates."""
    r = np.sqrt(rho ** 2 + z ** 2)
    theta = np.arccos(z / r)

    return r, theta


def calculate_beta_parameters(rho: np.ndarray, z: np.ndarray, lambda_beta: int) -> float:
    """Calculate the β parameters for given radius vector and angle."""
    # Calculate radius vector and angle
    r_beta, theta_beta = calculate_radius_vector(rho, z)

    # Calculate Y_l0 for the given points
    Y_lm = np.real(sph_harm(0, lambda_beta, 0, theta_beta))

    # Calculate Y_00 (constant for normalization)
    Y_00 = np.real(sph_harm(0, 0, 0, theta_beta))

    # Prepare integrand: R(θ,φ) * Y_lm * sin(θ)
    integrand_num = r_beta * Y_lm * np.sin(theta_beta)
    integrand_den = r_beta * Y_00 * np.sin(theta_beta)

    # Perform numerical integration over theta
    numerator = integrate.trapezoid(integrand_num, theta_beta)
    denominator = integrate.trapezoid(integrand_den, theta_beta)

    # Calculate the final coefficient using equation (8)
    beta_lm = 4 * np.pi * numerator / denominator

    # If beta_lm is close to zero, set it to zero
    if np.isclose(beta_lm, 0):
        beta_lm = 0

    print(f"β_{lambda_beta} = {beta_lm:.4f}")

    return beta_lm


@dataclass
class CassiniParameters:
    """Class to store Cassini shape parameters."""
    protons: int
    neutrons: int
    alpha: float = 0.0
    alpha_params: List[float] = field(default_factory=lambda: [0.0] * 5)  # Now includes α₂
    r0: float = 1.16  # Radius constant in fm

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not isinstance(self.alpha_params, list):
            raise TypeError("alpha_params must be a list")

        if len(self.alpha_params) != 5:  # Updated for 5 parameters
            _original_length = len(self.alpha_params)
            if len(self.alpha_params) < 5:
                self.alpha_params.extend([0.0] * (5 - len(self.alpha_params)))
            else:
                self.alpha_params = self.alpha_params[:5]

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons


class CassiniShapeCalculator:
    """Class for performing Cassini shape calculations."""

    def __init__(self, params: CassiniParameters):
        self.params = params

    def calculate_epsilon(self) -> float:
        """Calculate epsilon parameter from alpha and alpha parameters."""
        alpha = self.params.alpha
        alpha_params = self.params.alpha_params

        sum_all = sum(alpha_params)
        sum_alternating = sum((-1) ** n * val for n, val in enumerate(alpha_params, 1))

        # Calculate factorial sum term
        sum_factorial = 0
        for n in range(1, 3):  # For α₂ and α₄
            idx = 2 * n - 1  # Convert to 0-based index
            if idx < len(alpha_params):
                val = alpha_params[idx]
                sum_factorial += ((-1) ** n * val *
                                  double_factorial(2 * n - 1) /
                                  (2 ** n * math.factorial(n)))

        epsilon = ((alpha - 1) / 4 * ((1 + sum_all) ** 2 + (1 + sum_alternating) ** 2) +
                   (alpha + 1) / 2 * (1 + sum_factorial) ** 2)

        return epsilon

    def calculate_coordinates(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cylindrical coordinates using Cassini oval parametrization."""
        R_0 = self.params.r0 * (self.params.nucleons ** (1 / 3))
        epsilon = self.calculate_epsilon()
        s = epsilon * R_0 ** 2

        # Calculate R(x) using Legendre polynomials
        R = R_0 * (1 + sum(alpha_n * np.polynomial.legendre.Legendre.basis(n + 1)(x)
                           for n, alpha_n in enumerate(self.params.alpha_params)))

        # Calculate p(x)
        p2 = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2
        p = np.sqrt(p2)

        # Calculate ρ and z
        rho = np.sqrt(np.maximum(0, p - R ** 2 * (2 * x ** 2 - 1) - s)) / np.sqrt(2)
        z = np.sign(x) * np.sqrt(np.maximum(0, p + R ** 2 * (2 * x ** 2 - 1) + s)) / np.sqrt(2)

        return rho, z

    @staticmethod
    def calculate_zcm(rho: np.ndarray, z: np.ndarray) -> float:
        """Calculate the z-coordinate of the center of mass for given shape coordinates.

        Args:
            rho: Array of radial coordinates
            z: Array of vertical coordinates

        Returns:
            float: Z-coordinate of the center of mass
        """
        # Calculate differential elements
        dz = np.diff(z)
        rho_midpoints = (rho[1:] + rho[:-1]) / 2
        z_midpoints = (z[1:] + z[:-1]) / 2

        # Volume element dV = πρ²dz for constant density
        volume_elements = rho_midpoints * rho_midpoints * dz

        # Calculate center of mass
        total_volume = np.sum(volume_elements)
        z_cm = np.sum(volume_elements * z_midpoints) / total_volume

        return z_cm

    @staticmethod
    def integrate_volume(rho: np.ndarray, z: np.ndarray) -> float:
        """Calculate volume by numerically integrating the provided shape coordinates.

        Args:
            rho: Array of radial coordinates
            z: Array of vertical coordinates

        Returns:
            float: Volume calculated by numerical integration
        """
        # Calculate differential elements
        dz = np.diff(z)
        rho_midpoints = (rho[1:] + rho[:-1]) / 2

        # Volume element dV = πρ²dz
        volume_elements = np.pi * rho_midpoints * rho_midpoints * dz
        total_volume = np.abs(np.sum(volume_elements))

        return total_volume

    def calculate_sphere_volume(self) -> float:
        """Calculate the volume of a sphere with the same number of nucleons."""
        R_0 = self.params.r0 * (self.params.nucleons ** (1 / 3))
        return (4 / 3) * np.pi * R_0 ** 3


class CassiniShapePlotter:
    """Class for handling the plotting interface and user interaction."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Define all instance attributes
        self.line_polar = None
        self.line_polar_mirror = None

        self.initial_z = 92  # Uranium
        self.initial_n = 144
        self.initial_alpha = 0.0
        self.initial_alphas = [0.0, 0.0, 0.0, 0.0, 0.0]  # α₁, α₂, α₃, α₄
        self.x_points = np.linspace(-1, 1, 2000)

        # UI elements
        self.fig = None
        self.ax_plot = None
        self.line = None
        self.line_mirror = None
        self.line_unscaled = None
        self.line_unscaled_mirror = None
        self.sphere_line = None
        self.point_zcm = None
        self.point_zcm_bar = None
        self.slider_z = None
        self.slider_n = None
        self.btn_z_increase = None
        self.btn_z_decrease = None
        self.btn_n_increase = None
        self.btn_n_decrease = None
        self.slider_alpha = None
        self.btn_alpha_decrease = None
        self.btn_alpha_increase = None
        self.sliders = []
        self.buttons = []  # Will store alpha parameter +/- buttons
        self.reset_button = None
        self.save_button = None
        self.config_buttons = []  # Store configuration buttons

        # Initialize nuclear parameters
        self.nuclear_params = CassiniParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            alpha=self.initial_alpha,
            alpha_params=self.initial_alphas
        )

        # Set up the interface
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_plot = self.fig.add_subplot(111)

        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True)
        self.ax_plot.set_title('Nuclear Shape (Cassini Parametrization)', fontsize=14)
        self.ax_plot.set_xlabel('X (fm)', fontsize=12)
        self.ax_plot.set_ylabel('Y (fm)', fontsize=12)

        # Initialize the shape plot
        calculator = CassiniShapeCalculator(self.nuclear_params)
        rho_bar, z_bar = calculator.calculate_coordinates(self.x_points)

        # Calculate volume fixing factor
        volume_fixing_factor = calculator.calculate_sphere_volume() / calculator.integrate_volume(rho_bar, z_bar)

        # Calculate center of mass
        z_cm_bar = calculator.calculate_zcm(rho_bar, z_bar)

        # Transform rho_bar and z_bar to rho and z
        rho = rho_bar / volume_fixing_factor  # Scale the shape
        z = (z_bar - z_cm_bar) / volume_fixing_factor  # Center the shape

        # Calculate center of mass for both shapes
        z_cm = calculator.calculate_zcm(rho, z)
        z_cm_bar = calculator.calculate_zcm(rho_bar, z_bar)

        # Create reference sphere
        R_0 = self.nuclear_params.r0 * (self.nuclear_params.nucleons ** (1 / 3))
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = R_0 * np.cos(theta)
        sphere_y = R_0 * np.sin(theta)

        self.line, = self.ax_plot.plot(z, rho, 'b-', label='Scaled')
        self.line_mirror, = self.ax_plot.plot(z, -rho, 'b-')
        self.line_unscaled, = self.ax_plot.plot(z_bar, rho_bar, 'r--', label='Unscaled', alpha=0.5)
        self.line_unscaled_mirror, = self.ax_plot.plot(z_bar, -rho_bar, 'r--', alpha=0.5)
        self.sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray', alpha=0.5, label='R₀')
        self.point_zcm, = self.ax_plot.plot(z_cm, 0, 'bo', label='z_cm', markersize=8)
        self.point_zcm_bar, = self.ax_plot.plot(z_cm_bar, 0, 'ro', label='z_cm_bar', markersize=8)
        self.ax_plot.legend()

    def setup_controls(self):
        """Set up all UI controls."""
        # Create proton (Z) controls
        first_slider_y = 0.02
        ax_z = plt.axes((0.25, first_slider_y, 0.5, 0.02))
        ax_z_decrease = plt.axes((0.16, first_slider_y, 0.04, 0.02))
        ax_z_increase = plt.axes((0.80, first_slider_y, 0.04, 0.02))

        self.slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120,
                               valinit=self.initial_z, valstep=1)
        self.btn_z_decrease = Button(ax_z_decrease, '-')
        self.btn_z_increase = Button(ax_z_increase, '+')

        # Create neutron (N) controls
        ax_n = plt.axes((0.25, first_slider_y + 0.02, 0.5, 0.02))
        ax_n_decrease = plt.axes((0.16, 0.04, 0.04, 0.02))
        ax_n_increase = plt.axes((0.80, 0.04, 0.04, 0.02))

        self.slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180,
                               valinit=self.initial_n, valstep=1)
        self.btn_n_decrease = Button(ax_n_decrease, '-')
        self.btn_n_increase = Button(ax_n_increase, '+')

        # Create slider and buttons for main alpha parameter
        ax_alpha = plt.axes((0.25, first_slider_y + 0.04, 0.5, 0.02))
        ax_alpha_decrease = plt.axes((0.16, first_slider_y + 0.04, 0.04, 0.02))
        ax_alpha_increase = plt.axes((0.80, first_slider_y + 0.04, 0.04, 0.02))

        self.slider_alpha = Slider(ax=ax_alpha, label='α',
                                   valmin=0.0, valmax=1.05,
                                   valinit=self.initial_alpha, valstep=0.025)
        self.btn_alpha_decrease = Button(ax_alpha_decrease, '-')
        self.btn_alpha_increase = Button(ax_alpha_increase, '+')

        # Create sliders and buttons for alpha parameters with appropriate ranges
        param_ranges = [
            ('α₁', -1.0, 1.0),
            ('α₂', -1.0, 1.0),
            ('α₃', -1.0, 1.0),
            ('α₄', -1.0, 1.0)
        ]

        for i, (label, min_val, max_val) in enumerate(param_ranges):
            # Create slider
            ax = plt.axes((0.25, first_slider_y + 0.06 + i * 0.02, 0.5, 0.02))
            slider = Slider(ax=ax, label=label,
                            valmin=min_val, valmax=max_val,
                            valinit=self.initial_alphas[i], valstep=0.025)
            self.sliders.append(slider)

            # Create decrease/increase buttons
            ax_decrease = plt.axes((0.16, first_slider_y + 0.06 + i * 0.02, 0.04, 0.02))
            ax_increase = plt.axes((0.80, first_slider_y + 0.06 + i * 0.02, 0.04, 0.02))
            btn_decrease = Button(ax_decrease, '-')
            btn_increase = Button(ax_increase, '+')
            self.buttons.extend([btn_decrease, btn_increase])

        # Style font sizes for all sliders
        for slider in [self.slider_z, self.slider_n, self.slider_alpha] + self.sliders:
            slider.label.set_fontsize(12)
            slider.valtext.set_fontsize(12)

        # Create buttons
        ax_reset = plt.axes((0.8, 0.25, 0.1, 0.04))
        self.reset_button = Button(ax=ax_reset, label='Reset')

        ax_save = plt.axes((0.8, 0.2, 0.1, 0.04))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        # Create configuration buttons on the left
        config_labels = ['Ground State', 'First Saddle Point', 'Secondary Minimum', 'Second Saddle Point']
        for i, label in enumerate(config_labels):
            ax_config = plt.axes((0.02, 0.6 - i * 0.1, 0.1, 0.04))
            btn = Button(ax=ax_config, label=label)
            self.config_buttons.append(btn)

    def apply_configuration(self, config_num):
        """Apply a predefined configuration."""
        # These are placeholder values - you can set your own configurations
        configs = {
            0: {'Z': 92, 'N': 144, 'alpha': 0.250, 'params': [0.0, 0.0, 0.0, 0.075]},  # Ground state
            1: {'Z': 92, 'N': 144, 'alpha': 0.350, 'params': [0.0, 0.0, 0.0, -0.075]},  # First saddle point
            2: {'Z': 92, 'N': 144, 'alpha': 0.525, 'params': [0.0, 0.0, 0.0, 0.025]},  # Secondary minimum
            3: {'Z': 92, 'N': 144, 'alpha': 0.650, 'params': [0.2, 0.0, 0.025, 0.050]}  # Second saddle point
        }

        config = configs[config_num]
        self.slider_z.set_val(config['Z'])
        self.slider_n.set_val(config['N'])
        self.slider_alpha.set_val(config['alpha'])
        for slider, value in zip(self.sliders, config['params']):
            slider.set_val(value)

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)
        self.slider_alpha.on_changed(self.update_plot)
        for slider in self.sliders:
            slider.on_changed(self.update_plot)

        # Connect proton/neutron button handlers
        self.btn_z_decrease.on_clicked(self.create_button_handler(self.slider_z, -1))
        self.btn_z_increase.on_clicked(self.create_button_handler(self.slider_z, 1))
        self.btn_n_decrease.on_clicked(self.create_button_handler(self.slider_n, -1))
        self.btn_n_increase.on_clicked(self.create_button_handler(self.slider_n, 1))

        # Connect main alpha button handlers
        self.btn_alpha_decrease.on_clicked(self.create_button_handler(self.slider_alpha, -1))
        self.btn_alpha_increase.on_clicked(self.create_button_handler(self.slider_alpha, 1))

        # Connect alpha parameter button handlers
        for i, slider in enumerate(self.sliders):
            self.buttons[i * 2].on_clicked(self.create_button_handler(slider, -1))  # Decrease button
            self.buttons[i * 2 + 1].on_clicked(self.create_button_handler(slider, 1))  # Increase button

        # Connect action buttons
        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)

        # Connect configuration buttons
        for i, btn in enumerate(self.config_buttons):
            btn.on_clicked(lambda event, num=i: self.apply_configuration(num))

    @staticmethod
    def create_button_handler(slider_obj: Slider, increment: int):
        """Create a button click handler for a slider object."""

        def handler(_):
            """Handle button click event."""
            new_val = slider_obj.val + increment * slider_obj.valstep
            if slider_obj.valmin <= new_val <= slider_obj.valmax:
                slider_obj.set_val(new_val)

        return handler

    def reset_values(self, _):
        """Reset all sliders to their initial values."""
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.slider_alpha.set_val(self.initial_alpha)
        for slider, init_val in zip(self.sliders, self.initial_alphas):
            slider.set_val(init_val)

    def save_plot(self, _):
        """Save the current plot to a file."""
        number_of_protons = int(self.slider_z.val)
        number_of_neutrons = int(self.slider_n.val)
        params = [self.slider_alpha.val] + [s.val for s in self.sliders]
        filename = f"cassini_shape_{number_of_protons}_{number_of_neutrons}_{'_'.join(f'{p:.2f}' for p in params)}.png"
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self, _):
        """Update the plot with new parameters."""
        # Get current parameters
        current_params = CassiniParameters(
            protons=int(self.slider_z.val),
            neutrons=int(self.slider_n.val),
            alpha=self.slider_alpha.val,
            alpha_params=[s.val for s in self.sliders]
        )

        # Clear old polar plot if it exists
        if self.line_polar is not None:
            self.line_polar.remove()
            self.line_polar_mirror.remove()

        # Calculate new shape
        calculator = CassiniShapeCalculator(current_params)
        rho_bar, z_bar = calculator.calculate_coordinates(self.x_points)

        # Calculate the volume of the shape before scaling, sphere volume and volume scaling factor
        sphere_volume = calculator.calculate_sphere_volume()
        volume_pre_scale = calculator.integrate_volume(rho_bar, z_bar)
        volume_fixing_factor = sphere_volume / volume_pre_scale

        # Calculate the radius scaling factor
        radius_scaling_factor = (volume_fixing_factor ** (1 / 3))

        # Calculate c_male
        c_male = 1 / radius_scaling_factor

        # Calculate center of mass
        z_cm_bar = calculator.calculate_zcm(rho_bar, z_bar)

        # Transform rho_bar and z_bar to rho and z
        rho = rho_bar / c_male  # Scale the shape
        z = (z_bar - z_cm_bar) / c_male  # Center the shape

        # Calculate post-scale volume using the same integration method
        volume_post_scale = calculator.integrate_volume(rho, z)

        # Recalculate center of mass
        z_cm = calculator.calculate_zcm(rho, z)

        # Update plot for both scaled and unscaled shapes
        self.line.set_data(z, rho)
        self.line_mirror.set_data(z, -rho)
        self.line_unscaled.set_data(z_bar, rho_bar)
        self.line_unscaled_mirror.set_data(z_bar, -rho_bar)
        self.point_zcm.set_data([z_cm], [0])
        self.point_zcm_bar.set_data([z_cm_bar], [0])

        # Update reference sphere
        R_0 = current_params.r0 * (current_params.nucleons ** (1 / 3))
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = R_0 * np.cos(theta)
        sphere_y = R_0 * np.sin(theta)
        self.sphere_line.set_data(sphere_x, sphere_y)

        # Update plot limits
        max_val = max(np.max(np.abs(z)), np.max(np.abs(rho))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Calculate maximum dimensions
        max_x = np.max(np.abs(z))  # Maximum in x-direction (z-coordinate)
        max_y = np.max(np.abs(rho))  # Maximum in y-direction (rho-coordinate)
        total_length = 2 * max_x  # Full length in x-direction
        total_width = 2 * max_y  # Full width in y-direction

        # Calculate beta parameters for the shape
        for lambda_beta in range(1, 13):
            calculate_beta_parameters(rho, z, lambda_beta)

        # Calculate r and theta coordinates
        r, theta = calculate_radius_vector(rho, z)
        x_polar = r * np.cos(theta)
        y_polar = r * np.sin(theta)

        # Update the plot with the new shape
        self.line_polar, = self.ax_plot.plot(x_polar, y_polar, 'y', label='Polar', alpha=0.7)
        self.line_polar_mirror, = self.ax_plot.plot(x_polar, -y_polar, 'y', alpha=0.7)

        # Add volume, center of mass, and dimension information
        info_text = (
            f"Sphere volume: {sphere_volume:.4f} fm³\n"
            f"Shape volume (before scaling): {volume_pre_scale:.4f} fm³\n"
            f"Volume fixing factor: {volume_fixing_factor:.4f}\n"
            f"Radius scaling factor: {radius_scaling_factor:.4f}\n"
            f"c_male: {c_male:.4f}\n"
            f"Shape volume (after scaling): {volume_post_scale:.4f} fm³\n"
            f"Volume difference: {abs(sphere_volume - volume_post_scale):.4f} fm³\n"
            f"Z_bar center of mass: {z_cm_bar:.4f} fm\n"
            f"Z center of mass: {z_cm:.2f} fm\n"
            f"Max X length: {total_length:.4f} fm\n"
            f"Max Y length: {total_width:.4f} fm"
        )

        # Remove old text if it exists
        for artist in self.ax_plot.texts:
            artist.remove()

        # Add new text
        self.ax_plot.text(1.1 * max_val, 0.5 * max_val, info_text,
                          fontsize=24, verticalalignment='center')

        # Update title with current nuclear information
        self.ax_plot.set_title(f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, A={current_params.nucleons})',
                               fontsize=24)

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot(None)
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    plotter = CassiniShapePlotter()
    plotter.run()


if __name__ == '__main__':
    main()
