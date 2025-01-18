"""
Nuclear Shape Plotter using Cassini Ovals - A program to visualize and analyze nuclear shapes.
This version implements an object-oriented design with integrated beta parametrization.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy import integrate
from scipy.special import sph_harm_y

matplotlib.use('TkAgg')


def double_factorial(n):
    """Calculate double factorial n!!"""
    return math.prod(range(n, 0, -2))


@dataclass
class BetaShape:
    """Data class to store beta shape parameters and related data."""
    protons: int
    neutrons: int
    beta_parameters: List[float]  # β₁ through β₁₂
    radius: np.ndarray = None
    theta: np.ndarray = None

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons


class BetaParametrization:
    """Class for handling beta parametrization calculations and fitting."""

    def __init__(self, r0: float = 1.16):
        """Initialize BetaParametrization with radius constant.

        Args:
            r0: Radius constant in fm (default: 1.16)
        """
        self.r0 = r0

    def calculate_beta_parameters(self, rho: np.ndarray, z: np.ndarray) -> List[float]:
        """Calculate the first 12 β parameters for given shape coordinates.

        Args:
            rho: Array of radial coordinates
            z: Array of vertical coordinates

        Returns:
            List[float]: First 12 β parameters (β₁ through β₁₂)
        """
        # Calculate radius vector and angle
        r_beta, theta_beta = self._calculate_radius_vector(rho, z)

        # Calculate Y_00 for normalization
        Y_00 = np.real(sph_harm_y(0, 0, 0.0, 0.0))
        denominator = integrate.trapezoid(r_beta * Y_00 * np.sin(theta_beta), theta_beta)

        # Calculate all beta parameters
        betas = []
        for lambda_beta in range(1, 13):  # Calculate β₁ through β₁₂
            # Calculate Y_l0 for current lambda
            Y_lm = np.real(sph_harm_y(lambda_beta, 0, theta_beta, 0.0))

            # Calculate numerator integral
            integrand_num = r_beta * Y_lm * np.sin(theta_beta)
            numerator = integrate.trapezoid(integrand_num, theta_beta)

            # Calculate and round beta parameter
            beta_lm = float(np.sqrt(4 * np.pi) * numerator / denominator)
            beta_lm = np.round(beta_lm, 3)
            betas.append(beta_lm)

        return betas

    def calculate_beta_radius(self, shape: BetaShape, theta: np.ndarray) -> np.ndarray:
        """Calculate nuclear radius as a function of angle using beta parameters.

        Args:
            shape: BetaShape instance containing nuclear parameters
            theta: Array of angles to calculate radius at

        Returns:
            np.ndarray: Radius values at given angles
        """
        radius = np.ones_like(theta)

        # Add contributions from each harmonic
        for harmonic_index in range(1, 13):
            harmonic = np.real(sph_harm_y(harmonic_index, 0, theta, 0.0))
            radius += shape.beta_parameters[harmonic_index - 1] * harmonic

        # Scale by A^(1/3)
        return self.r0 * (shape.nucleons ** (1 / 3)) * radius

    def calculate_volume(self, shape: BetaShape, n_theta: int = 400) -> float:
        """Calculate nucleus volume by numerical integration.

        Args:
            shape: BetaShape instance containing nuclear parameters
            n_theta: Number of points for theta discretization

        Returns:
            float: Volume of the nucleus in fm³
        """
        theta = np.linspace(0, np.pi, n_theta)
        r = self.calculate_beta_radius(shape, theta)
        integrand = 2 * np.pi * (r ** 3 * np.sin(theta)) / 3

        return integrate.trapezoid(integrand, theta)

    def calculate_volume_analytical(self, shape: BetaShape) -> float:
        """Calculate the volume of a deformed nucleus using analytical formula.

        Args:
            shape: BetaShape instance containing nuclear parameters

        Returns:
            float: Volume calculated analytically in fm³
        """
        """Calculate the volume of a deformed nucleus using analytical formula."""
        nucleons = shape.nucleons
        r0 = self.r0
        beta10, beta100, beta110, beta120, beta20, beta30, beta40, beta50, beta60, beta70, beta80, beta90 = shape.beta_parameters

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

    def validate_fit(self, rho: np.ndarray, z: np.ndarray, shape: BetaShape) -> float:
        """Validate the beta parameter fit by comparing calculated and fitted shapes.

        Args:
            rho: Array of radial coordinates
            z: Array of vertical coordinates
            shape: BetaShape instance containing fitted parameters

        Returns:
            float: RMS error of the fit
        """
        r_alfa = np.sqrt(rho ** 2 + z ** 2)
        theta_alfa = np.arccos(z / r_alfa)

        # Calculate radius using beta parameters
        r_beta = self.calculate_beta_radius(shape, theta_alfa)

        # Calculate RMS error
        rms_error = np.sqrt(np.mean((r_alfa - r_beta) ** 2))

        return rms_error

    @staticmethod
    def _calculate_radius_vector(rho: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the radius vector for given shape in cylindrical coordinates.

        Args:
            rho: Array of radial coordinates
            z: Array of vertical coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]: Radius and theta arrays
        """
        r = np.sqrt(rho ** 2 + z ** 2)
        theta = np.arccos(z / r)
        return r, theta


@dataclass
class CassiniParameters:
    """Class to store Cassini shape parameters."""
    protons: int
    neutrons: int
    alpha: float = 0.0
    alpha_params: List[float] = field(default_factory=lambda: [0.0] * 5)
    r0: float = 1.16  # Radius constant in fm

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not isinstance(self.alpha_params, list):
            raise TypeError("alpha_params must be a list")

        if len(self.alpha_params) != 5:
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

        sum_factorial = 0
        for n in range(1, 3):
            idx = 2 * n - 1
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

        R = R_0 * (1 + sum(alpha_n * np.polynomial.legendre.Legendre.basis(n + 1)(x)
                           for n, alpha_n in enumerate(self.params.alpha_params)))

        p2 = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2
        p = np.sqrt(p2)

        rho = np.sqrt(np.maximum(0, p - R ** 2 * (2 * x ** 2 - 1) - s)) / np.sqrt(2)
        z = np.sign(x) * np.sqrt(np.maximum(0, p + R ** 2 * (2 * x ** 2 - 1) + s)) / np.sqrt(2)

        return rho, z

    @staticmethod
    def calculate_zcm(rho: np.ndarray, z: np.ndarray) -> float:
        """Calculate the z-coordinate of the center of mass."""
        dz = np.diff(z)
        rho_midpoints = (rho[1:] + rho[:-1]) / 2
        z_midpoints = (z[1:] + z[:-1]) / 2

        volume_elements = rho_midpoints * rho_midpoints * dz
        total_volume = np.sum(volume_elements)
        z_cm = np.sum(volume_elements * z_midpoints) / total_volume

        return z_cm

    @staticmethod
    def integrate_volume(rho: np.ndarray, z: np.ndarray) -> float:
        """Calculate volume by numerical integration."""
        dz = np.diff(z)
        rho_midpoints = (rho[1:] + rho[:-1]) / 2

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
        self.initial_z = 92  # Uranium
        self.initial_n = 144
        self.initial_alpha = 0.0
        self.initial_alphas = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.x_points = np.linspace(-1, 1, 2000)

        # Initialize beta parametrization
        self.beta_parametrization = BetaParametrization()

        # Initialize UI elements
        self.btn_alpha_increase = None
        self.btn_alpha_decrease = None
        self.btn_n_increase = None
        self.btn_n_decrease = None
        self.btn_z_increase = None
        self.btn_z_decrease = None
        self.fig = None
        self.ax_plot = None
        self.line = None
        self.line_mirror = None
        self.line_unscaled = None
        self.line_unscaled_mirror = None
        self.sphere_line = None
        self.point_zcm = None
        self.point_zcm_bar = None
        self.line_beta = None
        self.slider_z = None
        self.slider_n = None
        self.slider_alpha = None
        self.sliders = []
        self.buttons = []
        self.reset_button = None
        self.save_button = None
        self.config_buttons = []

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

        self.line, = self.ax_plot.plot(z, rho, 'b-', label='Scaled', linewidth=3, alpha=0.5)
        self.line_mirror, = self.ax_plot.plot(z, -rho, 'b-', linewidth=3, alpha=0.5)
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

        # Calculate new shape
        calculator = CassiniShapeCalculator(current_params)
        rho_bar, z_bar = calculator.calculate_coordinates(self.x_points)

        # Calculate volumes and scaling factors
        sphere_volume = calculator.calculate_sphere_volume()
        volume_pre_scale = calculator.integrate_volume(rho_bar, z_bar)
        volume_fixing_factor = sphere_volume / volume_pre_scale
        radius_scaling_factor = (volume_fixing_factor ** (1 / 3))
        c_male = 1 / radius_scaling_factor

        # Calculate center of mass and transform coordinates
        z_cm_bar = calculator.calculate_zcm(rho_bar, z_bar)
        rho = rho_bar / c_male
        z = (z_bar - z_cm_bar) / c_male
        volume_post_scale = calculator.integrate_volume(rho, z)
        z_cm = calculator.calculate_zcm(rho, z)

        # Create BetaShape instance and calculate parameters
        beta_shape = BetaShape(
            protons=current_params.protons,
            neutrons=current_params.neutrons,
            beta_parameters=self.beta_parametrization.calculate_beta_parameters(rho, z)
        )

        # Calculate beta volume
        beta_volume = self.beta_parametrization.calculate_volume(beta_shape)
        volume_analytical = self.beta_parametrization.calculate_volume_analytical(beta_shape)
        beta_volume_fixing_factor = sphere_volume / beta_volume
        beta_radius_fixing_factor = (beta_volume_fixing_factor ** (1 / 3))

        # Calculate beta-related quantities
        beta_theta = np.linspace(0, 2 * np.pi, 400)
        beta_radius = self.beta_parametrization.calculate_beta_radius(beta_shape, beta_theta) * beta_radius_fixing_factor
        beta_x = beta_radius * np.cos(beta_theta)
        beta_y = beta_radius * np.sin(beta_theta)

        # Calculate RMS error
        rms_error = self.beta_parametrization.validate_fit(rho, z, beta_shape)

        # Update plot
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

        # Update beta plot
        if self.line_beta is not None:
            self.line_beta.remove()
        self.line_beta, = self.ax_plot.plot(beta_x, beta_y, 'r+', label='Beta', alpha=1.0, markersize=1.5)

        # Update plot limits
        max_val = max(np.max(np.abs(z)), np.max(np.abs(rho))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Calculate dimensions
        max_x = np.abs(z[0] - z[-1])
        max_y = np.max(np.abs(rho))
        total_length = max_x
        total_width = 2 * max_y

        # Update information display
        info_text = (
                f"Sphere volume: {sphere_volume:.4f} fm³\n"
                f"Shape volume (before scaling): {volume_pre_scale:.4f} fm³\n"
                f"Volume fixing factor: {volume_fixing_factor:.4f}\n"
                f"Radius scaling factor: {radius_scaling_factor:.4f}\n"
                f"c_male: {c_male:.4f}\n"
                f"Shape volume (after scaling): {volume_post_scale:.4f} fm³\n"
                f"Volume difference: {abs(sphere_volume - volume_post_scale):.4f} fm³\n"
                f"Beta volume: {beta_volume:.4f} fm³\n"
                f"Analytical volume: {volume_analytical:.4f} fm³\n"
                f"Z_bar center of mass: {z_cm_bar:.4f} fm\n"
                f"Z center of mass: {z_cm:.2f} fm\n"
                f"Max X length: {total_length:.4f} fm\n"
                f"Max Y length: {total_width:.4f} fm\n"
                f"Beta parameters:\n" +
                ' '.join([f"β{i + 1}: {val:.4f}" for i, val in enumerate(beta_shape.beta_parameters[:4])]) + '\n' +
                ' '.join([f"β{i + 1}: {val:.4f}" for i, val in enumerate(beta_shape.beta_parameters[4:8], 4)]) + '\n' +
                ' '.join([f"β{i + 1}: {val:.4f}" for i, val in enumerate(beta_shape.beta_parameters[8:], 8)]) +
                f"\nRMS error: {rms_error:.4f}"
        )

        # Update display
        for artist in self.ax_plot.texts:
            artist.remove()
        self.ax_plot.text(1.1 * max_val, 0.15 * max_val, info_text,
                          fontsize=24, verticalalignment='center')
        self.ax_plot.set_title(
            f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, A={current_params.nucleons})',
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
