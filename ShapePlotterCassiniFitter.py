"""
Nuclear Shape Plotter - A program to visualize and analyze nuclear shapes using spherical harmonics.
This version implements an object-oriented design for better organization and maintainability.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox
from numpy import ndarray
from scipy import integrate
from scipy.special import sph_harm

matplotlib.use('TkAgg')


@dataclass
class NuclearParameters:
    """Class to store nuclear shape parameters."""
    protons: int
    neutrons: int
    beta_values: List[float] = field(default_factory=lambda: [0.0] * 12)
    r0: float = 1.16  # Radius constant in fm

    def __post_init__(self):
        """Validate and adjust beta_values after initialization."""
        if not isinstance(self.beta_values, list):
            raise TypeError("beta_values must be a list")

        if len(self.beta_values) != 12:
            original_length = len(self.beta_values)
            if len(self.beta_values) < 12:
                # Pad with zeros if list is too short
                self.beta_values.extend([0.0] * (12 - len(self.beta_values)))
                warnings.warn(f"beta_values list was too short (length {original_length}). Padded with zeros to length 12.")
            else:
                # Truncate if list is too long
                self.beta_values = self.beta_values[:12]
                warnings.warn(f"beta_values list was too long (length {original_length}). Truncated to length 12.")

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons


class NuclearShapeCalculator:
    """Class for performing nuclear shape calculations."""

    def __init__(self, params: NuclearParameters):
        self.params = params

    def calculate_sphere_volume(self) -> float:
        """Calculate the volume of a spherical nucleus."""
        return 4 / 3 * np.pi * self.params.nucleons * self.params.r0 ** 3

    def calculate_volume(self) -> float:
        """Calculate the volume of a deformed nucleus using analytical formula."""
        beta_params = tuple(self.params.beta_values)
        nucleons = self.params.nucleons
        r0 = self.params.r0

        beta10, beta20, beta30, beta40, beta50, beta60, beta70, beta80, beta90, beta100, beta110, beta120 = beta_params

        volume = (nucleons * r0 ** 3 * (74207381348100 * np.pi ** 1.5 + 648269351730 * np.sqrt(21) * beta100 ** 3 + 5807534192460 * np.sqrt(
            69) * beta10 * beta110 * beta120 + 8429951570040 * beta110 ** 2 * beta120 + 2724132411000 * beta120 ** 3 + 11131107202215 * np.sqrt(5) * beta10 ** 2 * beta20 + 6996695955678 * np.sqrt(
            5) * beta110 ** 2 * beta20 + 6990550416850 * np.sqrt(5) * beta120 ** 2 * beta20 + 2650263619575 * np.sqrt(5) * beta20 ** 3 + 2197030131010 * np.sqrt(161) * beta110 * beta120 * beta30 + 4770474515235 * np.sqrt(
            105) * beta10 * beta20 * beta30 + 7420738134810 * np.sqrt(
            5) * beta20 * beta30 ** 2 + 11968032555765 * beta110 ** 2 * beta40 + 11932146401175 * beta120 ** 2 * beta40 + 23852372576175 * beta20 ** 2 * beta40 + 10601054478300 * np.sqrt(
            21) * beta10 * beta30 * beta40 + 15178782548475 * beta30 ** 2 * beta40 + 7227991689750 * np.sqrt(5) * beta20 * beta40 ** 2 + 4503594822075 * beta40 ** 3 + 1395572678500 * np.sqrt(
            253) * beta110 * beta120 * beta50 + 2409330563250 * np.sqrt(385) * beta20 * beta30 * beta50 + 8432656971375 * np.sqrt(33) * beta10 * beta40 * beta50 + 3335996164500 * np.sqrt(
            77) * beta30 * beta40 * beta50 + 7135325129625 * np.sqrt(
            5) * beta20 * beta50 ** 2 + 12843585233325 * beta40 * beta50 ** 2 + 2832191612250 * np.sqrt(13) * beta110 ** 2 * beta60 + 2813654593750 * np.sqrt(13) * beta120 ** 2 * beta60 + 6486659208750 * np.sqrt(
            13) * beta30 ** 2 * beta60 + 5837993287875 * np.sqrt(65) * beta20 * beta40 * beta60 + 3891995525250 * np.sqrt(13) * beta40 ** 2 * beta60 + 2335197315150 * np.sqrt(429) * beta10 * beta50 * beta60 + 798726124350 * np.sqrt(
            3289) * beta110 * beta50 * beta60 + 908132289225 * np.sqrt(1001) * beta30 * beta50 * beta60 + 3357800061000 * np.sqrt(13) * beta50 ** 2 * beta60 + 22843567156410 * beta120 * beta60 ** 2 + 7083431855955 * np.sqrt(
            5) * beta20 * beta60 ** 2 + 12500173863450 * beta40 * beta60 ** 2 + 1044291885000 * np.sqrt(13) * beta60 ** 3 + 1042707290625 * np.sqrt(345) * beta110 * beta120 * beta70 + 2472247527750 * np.sqrt(
            345) * beta110 * beta40 * beta70 + 4540661446125 * np.sqrt(105) * beta30 * beta40 * beta70 + 3560036439960 * np.sqrt(165) * beta120 * beta50 * beta70 + 8173190603025 * np.sqrt(
            33) * beta20 * beta50 * beta70 + 2136781857000 * np.sqrt(165) * beta40 * beta50 * beta70 + 5993673108885 * np.sqrt(65) * beta10 * beta60 * beta70 + 383388539688 * np.sqrt(
            4485) * beta110 * beta60 * beta70 + 769241468520 * np.sqrt(
            1365) * beta30 * beta60 * beta70 + 506079913500 * np.sqrt(2145) * beta50 * beta60 * beta70 + 12690870642450 * beta120 * beta70 ** 2 + 7051380128100 * np.sqrt(
            5) * beta20 * beta70 ** 2 + 12297741898050 * beta40 * beta70 ** 2 + 3012380437500 * np.sqrt(13) * beta60 * beta70 ** 2 + 2238344983875 * np.sqrt(17) * beta110 ** 2 * beta80 + 2211803343750 * np.sqrt(
            17) * beta120 ** 2 * beta80 + 882945545625 * np.sqrt(2737) * beta110 * beta30 * beta80 + 11125113874875 * np.sqrt(17) * beta120 * beta40 * beta80 + 5609052374625 * np.sqrt(17) * beta40 ** 2 * beta80 + 395559604440 * np.sqrt(
            4301) * beta110 * beta50 * beta80 + 1282069114200 * np.sqrt(1309) * beta30 * beta50 * beta80 + 3247346111625 * np.sqrt(17) * beta50 ** 2 * beta80 + 1714091619240 * np.sqrt(
            221) * beta120 * beta60 * beta80 + 1410276025620 * np.sqrt(
            1105) * beta20 * beta60 * beta80 + 1821887688600 * np.sqrt(221) * beta40 * beta60 * beta80 + 2741266198125 * np.sqrt(17) * beta60 ** 2 * beta80 + 5238168095160 * np.sqrt(85) * beta10 * beta70 * beta80 + 276891723108 * np.sqrt(
            5865) * beta110 * beta70 * beta80 + 668025485820 * np.sqrt(1785) * beta30 * beta70 * beta80 + 433782783000 * np.sqrt(2805) * beta50 * beta70 * beta80 + 2521231453125 * np.sqrt(
            17) * beta70 ** 2 * beta80 + 10415266251390 * beta120 * beta80 ** 2 + 7030172969820 * np.sqrt(5) * beta20 * beta80 ** 2 + 12167607063150 * beta40 * beta80 ** 2 + 2939035522500 * np.sqrt(
            13) * beta60 * beta80 ** 2 + 800070781125 * np.sqrt(17) * beta80 ** 3 + 6111105 * beta100 ** 2 * (
                                                1440600 * beta120 + 31 * (36975 * np.sqrt(5) * beta20 + 63423 * beta40 + 15080 * np.sqrt(13) * beta60 + 12005 * np.sqrt(17) * beta80)) + 849332484000 * np.sqrt(
            437) * beta110 * beta120 * beta90 + 1000671618375 * np.sqrt(2185) * beta110 * beta20 * beta90 + 4002686473500 * np.sqrt(133) * beta120 * beta30 * beta90 + 1271441585700 * np.sqrt(
            437) * beta110 * beta40 * beta90 + 1785512103375 * np.sqrt(209) * beta120 * beta50 * beta90 + 3188303455050 * np.sqrt(209) * beta40 * beta50 * beta90 + 285681936540 * np.sqrt(
            5681) * beta110 * beta60 * beta90 + 1113375809700 * np.sqrt(1729) * beta30 * beta60 * beta90 + 506079913500 * np.sqrt(2717) * beta50 * beta60 * beta90 + 1241238758760 * np.sqrt(
            285) * beta120 * beta70 * beta90 + 6203093796900 * np.sqrt(57) * beta20 * beta70 * beta90 + 1590536871000 * np.sqrt(285) * beta40 * beta70 * beta90 + 363057329250 * np.sqrt(
            3705) * beta60 * beta70 * beta90 + 1550773449225 * np.sqrt(
            969) * beta10 * beta80 * beta90 + 222786443880 * np.sqrt(7429) * beta110 * beta80 * beta90 + 590770837800 * np.sqrt(2261) * beta30 * beta80 * beta90 + 380345773500 * np.sqrt(
            3553) * beta50 * beta80 * beta90 + 290445863400 * np.sqrt(
            4845) * beta70 * beta80 * beta90 + 9387573945750 * beta120 * beta90 ** 2 + 7015403698875 * np.sqrt(5) * beta20 * beta90 ** 2 + 12078695064150 * beta40 * beta90 ** 2 + 2890627878600 * np.sqrt(
            13) * beta60 * beta90 ** 2 + 2324911563975 * np.sqrt(17) * beta80 * beta90 ** 2 + 55655536011075 * np.sqrt(np.pi) * (beta10 ** 2 + beta100 ** 2 + beta110 ** 2 + beta120 ** 2 + beta20 ** 2 + beta30 ** 2 + beta40 ** 2 +
                                                                                                                                 beta50 ** 2 + beta60 ** 2 + beta70 ** 2 + beta80 ** 2 + beta90 ** 2) + 2035 * beta100 * (
                                                930397104 * np.sqrt(21) * beta110 ** 2 + 912234960 * np.sqrt(21) * beta120 ** 2 + 2242291194 * np.sqrt(105) * beta120 * beta20 + 2841109700 *
                                                np.sqrt(21) * beta120 * beta40 + 2462010390 * np.sqrt(21) * beta50 ** 2 + 635359725 * np.sqrt(273) * beta120 * beta60 + 1367783550 * np.sqrt(273) * beta40 * beta60 + 1391571090 *
                                                np.sqrt(21) * beta60 ** 2 + 10160677800 * np.sqrt(5) * beta30 * beta70 + 654157350 * np.sqrt(385) * beta50 * beta70 + 1156074444 * np.sqrt(21) * beta70 ** 2 + 491891400 *
                                                np.sqrt(357) * beta120 * beta80 + 544322025 * np.sqrt(1785) * beta20 * beta80 + 694207800 * np.sqrt(357) * beta40 * beta80 + 156997764 * np.sqrt(4641) * beta60 * beta80 + 1051409268 *
                                                np.sqrt(21) * beta80 ** 2 + 1822295475 * np.sqrt(57) * beta30 * beta90 + 166609872 * np.sqrt(4389) * beta50 * beta90 + 377957580 * np.sqrt(665) * beta70 * beta90 + 992760678 *
                                                np.sqrt(21) * beta90 ** 2 + 8940555 * beta10 * (209 * np.sqrt(161) * beta110 + 230 * np.sqrt(133) * beta90) + 858 * beta110 * (
                                                        1925658 * np.sqrt(69) * beta30 + 175305 * np.sqrt(5313) * beta50 + 394940 * np.sqrt(805) * beta70 + 108045 * np.sqrt(9177) * beta90)))) / (
                         5.5655536011075e13 * np.sqrt(np.pi))

        return volume

    def calculate_radius(self, theta: np.ndarray) -> np.ndarray:
        """Calculate nuclear radius as a function of polar angle theta."""
        radius = np.ones_like(theta)

        for harmonic_index in range(1, 13):
            harmonic = np.real(sph_harm(0, harmonic_index, 0, theta))
            radius += self.params.beta_values[harmonic_index - 1] * harmonic

        volume_fix = self.calculate_volume_fixing_factor() ** (1 / 3)
        return self.params.r0 * (self.params.nucleons ** (1 / 3)) * volume_fix * radius

    def calculate_volume_fixing_factor(self) -> float:
        """Calculate volume fixing factor to conserve volume."""
        initial_volume = self.calculate_volume()
        sphere_volume = self.calculate_sphere_volume()
        return sphere_volume / initial_volume

    def calculate_volume_by_integration(self, n_theta: int = 200, n_phi: int = 200) -> float:
        """Calculate nucleus volume by numerical integration."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)

        r = self.calculate_radius(theta_mesh)
        integrand = (r ** 3 * np.sin(theta_mesh)) / 3

        return integrate.trapezoid(integrate.trapezoid(integrand, theta, axis=1), phi)


class ShapeAnalyzer:
    """Class for analyzing nuclear shapes and finding key measurements."""

    def __init__(self, x_coords: np.ndarray, y_coords: np.ndarray, theta_vals: np.ndarray):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.theta_vals = theta_vals

    def find_neck_thickness(self, degree_range: Tuple[float, float]) -> tuple[float, ndarray[float, float], ndarray[float, float]]:
        """Find neck thickness between specified degree range."""
        start_rad, end_rad = np.radians(degree_range)
        mask = (self.theta_vals >= start_rad) & (self.theta_vals <= end_rad)

        relevant_x = self.x_coords[mask]
        relevant_y = self.y_coords[mask]

        distances = np.abs(relevant_y)
        neck_idx = np.argmin(distances)
        neck_thickness = distances[neck_idx] * 2

        return neck_thickness, relevant_x[neck_idx], relevant_y[neck_idx]

    @staticmethod
    def find_nearest_point(plot_x: np.ndarray, plot_y: np.ndarray, angle: float) -> tuple[ndarray[float, float], ndarray[float, float]]:
        """Find nearest point on curve to given angle."""
        angles = np.arctan2(plot_y, plot_x)
        angle_diff = np.abs(angles - angle)
        nearest_index = np.argmin(angle_diff)
        return plot_x[nearest_index], plot_y[nearest_index]


class NuclearShapePlotter:
    """Class for handling the plotting interface and user interaction."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        self.root = None
        self.submit_button = None
        self.text_box = None
        self.reset_button = None
        self.save_button = None
        self.slider_z = None
        self.slider_n = None
        self.btn_n_increase = None
        self.btn_n_decrease = None
        self.btn_z_increase = None
        self.btn_z_decrease = None
        self.volume_text = None
        self.line = None
        self.error_text = None
        self.ax_text = None
        self.ax_plot = None
        self.fig = None
        self.increase_buttons = None
        self.decrease_buttons = None
        self.sliders = None
        self.nuclear_params = None
        self.num_harmonics = 12
        self.theta = np.linspace(0, 2 * np.pi, 2000)
        self.initial_betas = [0.0] * self.num_harmonics
        self.initial_z = 102
        self.initial_n = 154
        self.setup_initial_parameters()
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def setup_initial_parameters(self):
        """Initialize default parameters."""
        self.num_harmonics = 12
        self.initial_z = 102
        self.initial_n = 154
        self.initial_betas = [0.0] * self.num_harmonics
        self.theta = np.linspace(0, 2 * np.pi, 2000)

        self.nuclear_params = NuclearParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            beta_values=self.initial_betas
        )

        self.sliders = []
        self.decrease_buttons = []
        self.increase_buttons = []

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(15, 8))
        self.ax_plot = self.fig.add_subplot(121)
        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.axis('off')

        plt.subplots_adjust(left=0.1, bottom=0.48, right=0.9, top=0.98)

        # Add keyboard input instructions
        self.ax_text.text(0.1, 0.25, 'Keyboard Input Format (works with Ctrl+V):\n'
                                     'Z N β10 β20 β30 β40 β50 β60 β70 β80 β90 β100 β110 β120\n'
                                     'Example: 102 154 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0',
                          fontsize=12, verticalalignment='top')

        # Add error message text (initially empty)
        self.error_text = self.ax_text.text(0.1, 0.15, '', color='red', fontsize=12,
                                            verticalalignment='top')

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True)
        self.ax_plot.set_title('Nuclear Shape with Volume Conservation', fontsize=18)
        self.ax_plot.set_xlabel('X (fm)', fontsize=18)
        self.ax_plot.set_ylabel('Y (fm)', fontsize=18)

        # Initialize the shape plot
        calculator = NuclearShapeCalculator(self.nuclear_params)
        radius = calculator.calculate_radius(self.theta)
        x = radius * np.sin(self.theta)
        y = radius * np.cos(self.theta)
        self.line, = self.ax_plot.plot(x, y)

        # Create a text box for volume information
        self.volume_text = self.ax_text.text(0.1, 0.25, '', fontsize=24)

    def setup_controls(self):
        """Set up all UI controls."""
        self.create_proton_neutron_controls()
        self.create_beta_controls()
        self.create_action_buttons()
        self.create_text_input()

    def create_proton_neutron_controls(self):
        """Create controls for proton and neutron numbers."""
        # Proton controls
        ax_z = plt.axes((0.25, 0.00, 0.5, 0.02))
        ax_z_decrease = plt.axes((0.16, 0.00, 0.04, 0.02))
        ax_z_increase = plt.axes((0.80, 0.00, 0.04, 0.02))

        self.slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120,
                               valinit=self.initial_z, valstep=1)
        self.btn_z_decrease = Button(ax_z_decrease, '-')
        self.btn_z_increase = Button(ax_z_increase, '+')

        # Neutron controls
        ax_n = plt.axes((0.25, 0.03, 0.5, 0.02))
        ax_n_decrease = plt.axes((0.16, 0.03, 0.04, 0.02))
        ax_n_increase = plt.axes((0.80, 0.03, 0.04, 0.02))

        self.slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180,
                               valinit=self.initial_n, valstep=1)
        self.btn_n_decrease = Button(ax_n_decrease, '-')
        self.btn_n_increase = Button(ax_n_increase, '+')

        # Style settings
        for slider in [self.slider_z, self.slider_n]:
            slider.label.set_fontsize(18)
            slider.valtext.set_fontsize(18)

    def create_beta_controls(self):
        """Create controls for beta parameters."""
        slider_height = 0.03

        for i in range(self.num_harmonics):
            ax_decrease = plt.axes((0.16, 0.06 + i * slider_height, 0.04, 0.02))
            ax_slider = plt.axes((0.25, 0.06 + i * slider_height, 0.5, 0.02))
            ax_increase = plt.axes((0.80, 0.06 + i * slider_height, 0.04, 0.02))

            valmin, valmax = (-1.6, 1.6) if i == 0 else (0.0, 3.0) if i == 1 else (-1.0, 1.0)

            slider = Slider(
                ax=ax_slider,
                label=f'β{i + 1}0',
                valmin=valmin,
                valmax=valmax,
                valinit=self.initial_betas[i],
                valstep=0.01
            )

            btn_decrease = Button(ax_decrease, '-')
            btn_increase = Button(ax_increase, '+')

            slider.label.set_fontsize(18)
            slider.valtext.set_fontsize(18)

            self.sliders.append(slider)
            self.decrease_buttons.append(btn_decrease)
            self.increase_buttons.append(btn_increase)

    def create_action_buttons(self):
        """Create save and reset buttons."""
        ax_save = plt.axes((0.75, 0.45, 0.1, 0.04))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        ax_reset = plt.axes((0.86, 0.45, 0.1, 0.04))
        self.reset_button = Button(ax=ax_reset, label='Reset')

    def create_text_input(self):
        """Create text input field and submit button."""
        ax_input = plt.axes((0.25, 0.42, 0.5, 0.02))
        self.text_box = TextBox(ax_input, 'Parameters')
        self.text_box.label.set_fontsize(12)

        ax_submit = plt.axes((0.80, 0.42, 0.1, 0.02))
        self.submit_button = Button(ax_submit, 'Submit')

        # Enable key events for the text box
        text_box_widget = self.text_box.ax.figure.canvas.get_tk_widget()
        self.root = text_box_widget.master
        self.root.bind_all('<Control-v>', self.handle_paste)

    def handle_paste(self, _):
        """Handle paste events from clipboard."""
        try:
            clipboard_text = self.root.clipboard_get()
            self.text_box.set_val(clipboard_text)
            return "break"  # Prevents default paste behavior
        except Exception as e:
            print(f"Error pasting from clipboard: {e}")
            return "break"

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        for slider in self.sliders:
            slider.on_changed(self.update_plot)
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)

        # Connect button handlers
        for i, slider in enumerate(self.sliders):
            self.decrease_buttons[i].on_clicked(self.create_button_handler(slider, -1))
            self.increase_buttons[i].on_clicked(self.create_button_handler(slider, 1))

        self.btn_z_decrease.on_clicked(self.create_button_handler(self.slider_z, -1))
        self.btn_z_increase.on_clicked(self.create_button_handler(self.slider_z, 1))
        self.btn_n_decrease.on_clicked(self.create_button_handler(self.slider_n, -1))
        self.btn_n_increase.on_clicked(self.create_button_handler(self.slider_n, 1))

        # Connect action buttons
        self.submit_button.on_clicked(self.submit_parameters)
        self.save_button.on_clicked(self.save_plot)
        self.reset_button.on_clicked(self.reset_values)

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
        for slider in self.sliders:
            slider.set_val(0.0)
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.text_box.set_val('')

    def submit_parameters(self, _):
        """Handle parameter submission from text input."""
        try:
            values = [float(val) for val in self.text_box.text.split()]
            if len(values) != 14:  # 2 for Z,N + 12 for betas
                raise ValueError("Expected 14 values: Z N β10 β20 β30 β40 β50 β60 β70 β80 β90 β100 β110 β120")

            # Validate Z and N ranges
            if not (82 <= values[0] <= 120 and 100 <= values[1] <= 180):
                raise ValueError("Z must be between 82-120 and N between 100-180")

            # Update Z and N sliders
            self.slider_z.set_val(int(values[0]))
            self.slider_n.set_val(int(values[1]))

            # Update beta parameter sliders
            for i, slider in enumerate(self.sliders):
                if not (slider.valmin <= values[i + 2] <= slider.valmax):
                    raise ValueError(f"β{i + 1}0 must be between {slider.valmin} and {slider.valmax}")
                slider.set_val(values[i + 2])

            # Clear the text box and error message
            self.text_box.set_val('')
            self.error_text.set_text('')
            self.fig.canvas.draw_idle()

        except (ValueError, IndexError) as e:
            error_msg = f"Error: {str(e)}\nPlease use format: Z N β10 β20 β30 β40 β50 β60 β70 β80"
            self.error_text.set_text(error_msg)
            self.fig.canvas.draw_idle()

    def save_plot(self, _=None):
        """Save the current plot to a file."""
        parameters = [s.val for s in self.sliders]
        number_of_protons = int(self.slider_z.val)
        number_of_neutrons = int(self.slider_n.val)
        beta_values = "_".join(f"{p:.2f}" for p in parameters)
        filename = f"{number_of_protons}_{number_of_neutrons}_{beta_values}.png"
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self, _=None):
        """Update the plot with new parameters and calculate measurements."""
        # Get current parameters
        current_params = NuclearParameters(
            protons=int(self.slider_z.val),
            neutrons=int(self.slider_n.val),
            beta_values=[s.val for s in self.sliders]
        )

        # Calculate new shape
        calculator = NuclearShapeCalculator(current_params)
        plot_radius = calculator.calculate_radius(self.theta)
        plot_x = plot_radius * np.cos(self.theta)
        plot_y = plot_radius * np.sin(self.theta)

        # Update plot data
        self.line.set_data(plot_x, plot_y)

        # Initialize shape analyzer
        analyzer = ShapeAnalyzer(plot_x, plot_y, self.theta)

        # Find intersection points with axes
        x_axis_positive = analyzer.find_nearest_point(plot_x, plot_y, 0)
        x_axis_negative = analyzer.find_nearest_point(plot_x, plot_y, np.pi)
        y_axis_positive = analyzer.find_nearest_point(plot_x, plot_y, np.pi / 2)
        y_axis_negative = analyzer.find_nearest_point(plot_x, plot_y, -np.pi / 2)

        # Remove previous lines if they exist
        for attr in ['x_axis_line', 'y_axis_line', 'neck_line']:
            if hasattr(self.ax_plot, attr):
                getattr(self.ax_plot, attr).remove()

        # Draw axis lines
        self.ax_plot.x_axis_line = self.ax_plot.plot(
            [x_axis_negative[0], x_axis_positive[0]],
            [x_axis_negative[1], x_axis_positive[1]],
            color='red'
        )[0]

        self.ax_plot.y_axis_line = self.ax_plot.plot(
            [y_axis_negative[0], y_axis_positive[0]],
            [y_axis_negative[1], y_axis_positive[1]],
            color='blue'
        )[0]

        # Calculate and draw necks
        neck_thickness_45_135, neck_x_45_135, neck_y_45_135 = analyzer.find_neck_thickness(
            (45, 135)
        )
        neck_thickness_30_150, neck_x_30_150, neck_y_30_150 = analyzer.find_neck_thickness(
            (30, 150)
        )

        # Remove previous neck lines
        for attr in ['neck_line_45_135', 'neck_line_30_150']:
            if hasattr(self.ax_plot, attr):
                getattr(self.ax_plot, attr).remove()

        # Draw neck lines
        self.ax_plot.neck_line_45_135 = self.ax_plot.plot(
            [neck_x_45_135, neck_x_45_135],
            [-neck_thickness_45_135 / 2, neck_thickness_45_135 / 2],
            color='green',
            linewidth=2,
            label='Neck (45-135)'
        )[0]

        self.ax_plot.neck_line_30_150 = self.ax_plot.plot(
            [neck_x_30_150, neck_x_30_150],
            [-neck_thickness_30_150 / 2, neck_thickness_30_150 / 2],
            color='purple',
            linewidth=2,
            label='Neck (30-150)'
        )[0]

        # Update plot limits
        max_radius = np.max(np.abs(plot_radius)) * 1.5
        self.ax_plot.set_xlim(-max_radius, max_radius)
        self.ax_plot.set_ylim(-max_radius, max_radius)

        # Calculate measurements
        max_x_length = np.max(plot_x) - np.min(plot_x)
        max_y_length = np.max(plot_y) - np.min(plot_y)
        along_x_length = (calculator.calculate_radius(np.array([0.0]))[0] + calculator.calculate_radius(np.array([np.pi]))[0])
        along_y_length = (calculator.calculate_radius(np.array([np.pi / 2]))[0] + calculator.calculate_radius(np.array([-np.pi / 2]))[0])

        # Calculate volumes
        sphere_volume = calculator.calculate_sphere_volume()
        shape_volume = calculator.calculate_volume()
        volume_fix = calculator.calculate_volume_fixing_factor()
        shape_volume_integration = calculator.calculate_volume_by_integration()

        # Check calculations
        volume_mismatch = abs(sphere_volume - shape_volume_integration) > 1.0
        negative_radius = np.any(plot_radius < 0)

        # Update information display
        self.volume_text.set_text(
            f'Sphere Volume: {sphere_volume:.4f} fm³\n'
            f'Shape Volume: {shape_volume:.4f} fm³\n'
            f'Volume Fixing Factor: {volume_fix:.8f}\n'
            f'Radius Fixing Factor: {volume_fix ** (1 / 3):.8f}\n'
            f'Integrated Volume: {shape_volume_integration:.4f} fm³\n'
            f'Volume Conservation: {"✓" if not volume_mismatch else f"✗: {sphere_volume:.4f} vs {shape_volume_integration:.4f} fm³"}\n'
            f'Max X Length: {max_x_length:.2f} fm\n'
            f'Max Y Length: {max_y_length:.2f} fm\n'
            f'Length Along X Axis (red): {along_x_length:.2f} fm\n'
            f'Length Along Y Axis (blue): {along_y_length:.2f} fm\n'
            f'Neck Thickness (45°-135°, green): {neck_thickness_45_135:.2f} fm\n'
            f'Neck Thickness (30°-150°, purple): {neck_thickness_30_150:.2f} fm\n' +
            ('Negative radius detected!\n' if negative_radius else '')
        )

        # Update the legend
        self.ax_plot.legend(fontsize='small', loc='upper right')
        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot()
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    plotter = NuclearShapePlotter()
    plotter.run()


if __name__ == '__main__':
    main()
