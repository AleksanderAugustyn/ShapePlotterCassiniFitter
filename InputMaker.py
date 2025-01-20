import itertools

import numpy as np
import pandas as pd

# Import necessary classes from the original file
from ShapePlotterCassiniFitter import (BetaParametrization, BetaShape, CassiniParameters, CassiniShapeCalculator)


def analyze_shape_parameters(
        protons: int = 92,
        neutrons: int = 144,
        alpha_range: np.ndarray = np.arange(0.0, 1.1, 0.1),
        alpha1_range: np.ndarray = np.arange(-0.2, 0.3, 0.1),
        alpha3_range: np.ndarray = np.arange(-0.2, 0.3, 0.1),
        alpha4_range: np.ndarray = np.arange(-0.2, 0.3, 0.1)
) -> pd.DataFrame:
    """
    Analyze nuclear shapes for different parameter combinations and calculate beta parameters.

    Args:
        protons: Number of protons (default: 92 for Uranium)
        neutrons: Number of neutrons (default: 144)
        alpha_range: Range of alpha values to analyze
        alpha1_range: Range of alpha1 values to analyze
        alpha3_range: Range of alpha3 values to analyze
        alpha4_range: Range of alpha4 values to analyze

    Returns:
        pd.DataFrame: Results containing parameters and analysis
    """
    # Initialize results storage
    results = []

    # Initialize tools
    beta_parametrization = BetaParametrization()
    x_points = np.linspace(-1, 1, 2000)

    # Calculate all parameter combinations
    param_combinations = itertools.product(
        alpha_range,
        alpha1_range,
        alpha3_range,
        alpha4_range
    )

    total_combinations = len(alpha_range) * len(alpha1_range) * len(alpha3_range) * len(alpha4_range)
    print(f"Analyzing {total_combinations} parameter combinations...")

    # Iterate through all combinations
    for combo_num, (alpha, alpha1, alpha3, alpha4) in enumerate(param_combinations, 1):
        if combo_num % 100 == 0:
            print(f"Processing combination {combo_num}/{total_combinations}")

        # Create parameter set with current values
        alpha_params = [alpha1, 0.0, alpha3, alpha4, 0.0]  # alpha2 and alpha5 fixed at 0
        params = CassiniParameters(
            protons=protons,
            neutrons=neutrons,
            alpha=alpha,
            alpha_params=alpha_params
        )

        # Calculate shape coordinates
        calculator = CassiniShapeCalculator(params)
        rho_bar, z_bar = calculator.calculate_coordinates(x_points)

        # Calculate volume fixing factor and transform coordinates
        sphere_volume = calculator.calculate_sphere_volume()
        volume_pre_scale = calculator.integrate_volume(rho_bar, z_bar)
        volume_fixing_factor = sphere_volume / volume_pre_scale
        z_cm_bar = calculator.calculate_zcm(rho_bar, z_bar)

        # Scale coordinates
        c_male = volume_fixing_factor ** (-1 / 3)
        rho = rho_bar / c_male
        z = (z_bar - z_cm_bar) / c_male

        # Calculate beta parameters
        beta_shape = BetaShape(
            protons=protons,
            neutrons=neutrons,
            beta_parameters=beta_parametrization.calculate_beta_parameters(rho, z)
        )

        # Calculate fit quality
        fit_metrics = beta_parametrization.validate_fit(rho, z, beta_shape)

        # Store results
        result = {
            'alpha': alpha,
            'alpha1': alpha1,
            'alpha3': alpha3,
            'alpha4': alpha4,
            'RMSE': fit_metrics['RMSE'],
            'MAE': fit_metrics['MAE'],
            'MAPE': fit_metrics['MAPE'],
            'R_squared': fit_metrics['R²']
        }

        # Add beta parameters to results
        for i, beta in enumerate(beta_shape.beta_parameters, 1):
            result[f'beta{i}'] = beta

        results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    return df


def main(protons: int = 92, neutrons: int = 144):
    """Main function to run the analysis and save results.
    
    Args:
        protons: Number of protons (default: 92 for Uranium)
        neutrons: Number of neutrons (default: 144)
    """
    # Define parameter ranges
    alpha_range = np.arange(0.0, 0.951, 0.025)
    alpha1_range = np.arange(-0.25, 0.251, 0.25)
    alpha3_range = np.arange(-0.25, 0.251, 0.25)
    alpha4_range = np.arange(-0.25, 0.251, 0.25)

    # Print the number of combinations
    total_combinations = len(alpha_range) * len(alpha1_range) * len(alpha3_range) * len(alpha4_range)
    print(f"Total combinations: {total_combinations}")

    # Run analysis
    results_df = analyze_shape_parameters(
        protons=protons,
        neutrons=neutrons,
        alpha_range=alpha_range,
        alpha1_range=alpha1_range,
        alpha3_range=alpha3_range,
        alpha4_range=alpha4_range
    )

    # Save detailed results to CSV
    output_filename = 'nuclear_shape_analysis.txt'
    # Add protons and neutrons columns
    results_df.insert(0, 'protons', protons)
    results_df.insert(1, 'neutrons', neutrons)
    
    # Reorder columns to have parameters in logical order
    cols_order = ['protons', 'neutrons', 
                  'alpha', 'alpha1', 'alpha3', 'alpha4',
                  'beta1', 'beta2', 'beta3', 'beta4', 'beta5',
                  'RMSE', 'MAE', 'MAPE', 'R_squared']
    results_df = results_df[cols_order]
    
    results_df.to_csv(output_filename, sep=' ', float_format='%.6f', header=True, index=False)
    print(f"\nResults saved to {output_filename}")

    # Create summary file
    best_fit = results_df.loc[results_df['RMSE'].idxmin()]
    worst_fit = results_df.loc[results_df['RMSE'].idxmax()]
    
    with open('analysis_summary.txt', 'w') as f:
        f.write("Nuclear Shape Analysis Summary\n")
        f.write("=============================\n\n")
        f.write(f"Nucleus: Z={protons}, N={neutrons}\n\n")
        
        f.write("RMSE Statistics:\n")
        f.write(f"Average: {results_df['RMSE'].mean():.6f} fm\n")
        f.write(f"Minimum: {results_df['RMSE'].min():.6f} fm\n")
        f.write(f"Maximum: {results_df['RMSE'].max():.6f} fm\n")
        f.write(f"Std Dev: {results_df['RMSE'].std():.6f} fm\n\n")
        
        f.write("Best Fit Parameters:\n")
        f.write(f"α: {best_fit['alpha']:.6f}\n")
        f.write(f"α₁: {best_fit['alpha1']:.6f}\n")
        f.write(f"α₃: {best_fit['alpha3']:.6f}\n")
        f.write(f"α₄: {best_fit['alpha4']:.6f}\n")
        f.write(f"β₁: {best_fit['beta1']:.6f}\n")
        f.write(f"β₂: {best_fit['beta2']:.6f}\n")
        f.write(f"β₃: {best_fit['beta3']:.6f}\n")
        f.write(f"β₄: {best_fit['beta4']:.6f}\n")
        f.write(f"β₅: {best_fit['beta5']:.6f}\n")
        f.write(f"RMSE: {best_fit['RMSE']:.6f} fm\n")

    print("\nSummary statistics saved to analysis_summary.txt")


if __name__ == '__main__':
    main()
