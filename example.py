"""
Example Usage Script
===================

This script demonstrates how to use the electricity market simulation.

"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Import from main module
from electricity_market import (
    MarketConfig, GeneratorConfig,
    run_simulation, plot_learning_curves, 
    plot_policy_analysis, create_summary_report
)

# Import scenarios
from market_scenarios import (
    create_market_scenarios,
    run_comparative_analysis,
    analyze_demand_elasticity,
    analyze_price_caps,
    quick_scenario_test
)


# ============================================================================
# EXAMPLE 1: Simple Duopoly Simulation
# ============================================================================

def example_1_simple_duopoly():
    """Run a simple symmetric duopoly simulation."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Duopoly Simulation")
    print("="*70)
    
    # Configure market
    market_config = MarketConfig(
        demand_intercept=1000.0,
        demand_slope=-10.0,
        demand_shock_std=50.0,
        price_cap=1000.0
    )
    
    # Configure generators (symmetric duopoly)
    generator_configs = [
        GeneratorConfig(id=1, capacity=400, marginal_cost=30, learning_rate=0.1),
        GeneratorConfig(id=2, capacity=400, marginal_cost=30, learning_rate=0.1)
    ]
    
    # Run simulation
    market, analyzer = run_simulation(
        market_config=market_config,
        generator_configs=generator_configs,
        n_episodes=10000,
        eval_interval=2500
    )
    
    # Generate outputs
    print("\nGenerating visualizations...")
    plot_learning_curves(market, save_path='example1_learning.png')
    plot_policy_analysis(market, save_path='example1_policy.png')
    
    print("\nGenerating report...")
    summary_df, gen_df = create_summary_report(market, analyzer, 
                                              save_path='example1_summary.txt')
    
    print("\n✓ Example 1 complete!")
    print("Files created:")
    print("  - example1_learning.png")
    print("  - example1_policy.png")
    print("  - example1_summary.txt")


# ============================================================================
# EXAMPLE 2: Compare Market Structures
# ============================================================================

def example_2_compare_scenarios():
    """Compare different market structures."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Compare Market Structures")
    print("="*70)
    
    # Quick test of 3 scenarios
    scenarios = create_market_scenarios()
    
    # Select scenarios to compare
    selected_scenarios = {
        'sym_duopoly': scenarios['sym_duopoly'],
        'sym_triopoly': scenarios['sym_triopoly'],
        'oligopoly_5': scenarios['oligopoly_5']
    }
    
    # Run comparative analysis
    results_df = run_comparative_analysis(
        scenarios=selected_scenarios,
        n_episodes=8000,
        n_runs=2
    )
    
    # Save results
    results_df.to_csv('example2_comparison.csv', index=False)
    
    print("\n✓ Example 2 complete!")
    print("File created: example2_comparison.csv")
    
    return results_df


# ============================================================================
# EXAMPLE 3: Sensitivity Analysis
# ============================================================================

def example_3_sensitivity():
    """Run sensitivity analysis."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Sensitivity Analysis")
    print("="*70)
    
    # Test demand elasticity
    print("\n--- Demand Elasticity ---")
    elasticity_results = analyze_demand_elasticity(
        base_scenario_key='sym_duopoly',
        elasticities=[-5.0, -10.0, -20.0, -30.0],
        n_episodes=6000,
        n_runs=2
    )
    elasticity_results.to_csv('example3_elasticity.csv', index=False)
    
    # Test price caps
    print("\n--- Price Caps ---")
    price_cap_results = analyze_price_caps(
        base_scenario_key='sym_duopoly',
        price_caps=[60.0, 80.0, 100.0, 150.0, 300.0],
        n_episodes=6000,
        n_runs=2
    )
    price_cap_results.to_csv('example3_price_caps.csv', index=False)
    
    print("\n✓ Example 3 complete!")
    print("Files created:")
    print("  - example3_elasticity.csv")
    print("  - example3_price_caps.csv")


# ============================================================================
# EXAMPLE 4: Custom Configuration
# ============================================================================

def example_4_custom():
    """Create and run a custom market configuration."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Market Configuration")
    print("="*70)
    
    # Custom market with more elastic demand and lower price cap
    market_config = MarketConfig(
        demand_intercept=1200.0,      # Higher base demand
        demand_slope=-15.0,           # More elastic
        demand_shock_std=80.0,        # More uncertainty
        price_cap=200.0,              # Lower cap (binding)
        bid_step=10.0                 # Coarser bids
    )
    
    # Asymmetric generators (different costs)
    generator_configs = [
        GeneratorConfig(id=1, capacity=500, marginal_cost=20, learning_rate=0.15),  # Efficient
        GeneratorConfig(id=2, capacity=400, marginal_cost=35, learning_rate=0.15),  # Less efficient
        GeneratorConfig(id=3, capacity=300, marginal_cost=40, learning_rate=0.15)   # Peaker
    ]
    
    # Run simulation
    market, analyzer = run_simulation(
        market_config=market_config,
        generator_configs=generator_configs,
        n_episodes=10000,
        eval_interval=2500
    )
    
    # Outputs
    plot_learning_curves(market, save_path='example4_learning.png')
    create_summary_report(market, analyzer, save_path='example4_summary.txt')
    
    print("\n✓ Example 4 complete!")
    print("Files created:")
    print("  - example4_learning.png")
    print("  - example4_summary.txt")


# ============================================================================
# EXAMPLE 5: Quick Test
# ============================================================================

def example_5_quick_test():
    """Quick test of predefined scenario."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Quick Scenario Test")
    print("="*70)
    
    # Test a scenario quickly
    market, analyzer = quick_scenario_test(
        scenario_key='high_concentration',
        n_episodes=5000
    )
    
    print("\n✓ Example 5 complete!")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu for running examples."""
    
    print("\n" + "="*70)
    print("ELECTRICITY MARKET MARL - EXAMPLE SCRIPTS")
    print("="*70)
    print("\nAvailable Examples:")
    print("  1. Simple Duopoly Simulation")
    print("  2. Compare Market Structures")
    print("  3. Sensitivity Analysis")
    print("  4. Custom Configuration")
    print("  5. Quick Scenario Test")
    print("  6. Run All Examples")
    print("  0. Exit")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    if choice == '1':
        example_1_simple_duopoly()
    elif choice == '2':
        example_2_compare_scenarios()
    elif choice == '3':
        example_3_sensitivity()
    elif choice == '4':
        example_4_custom()
    elif choice == '5':
        example_5_quick_test()
    elif choice == '6':
        print("\nRunning all examples (this will take a while)...")
        example_1_simple_duopoly()
        example_2_compare_scenarios()
        example_3_sensitivity()
        example_4_custom()
        example_5_quick_test()
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED!")
        print("="*70)
    elif choice == '0':
        print("\nExiting...")
    else:
        print("\nInvalid choice!")


# ============================================================================
# DIRECT RUN
# ============================================================================

if __name__ == "__main__":
    # If you want to run a specific example directly, uncomment one:
    
    # example_1_simple_duopoly()
    # example_2_compare_scenarios()
    # example_3_sensitivity()
    # example_4_custom()
    # example_5_quick_test()
    
    # Or run the interactive menu:
    main()
