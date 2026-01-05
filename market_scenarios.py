"""
Market Scenarios and Advanced Analysis
======================================

Predefined market scenarios and sensitivity analysis tools.

"""

import numpy as np
import pandas as pd
from typing import Dict, List
from electricity_market import (
    MarketConfig, GeneratorConfig, QLearningGenerator,
    ElectricityMarket, MarketAnalyzer, run_simulation
)


# ============================================================================
# PREDEFINED SCENARIOS
# ============================================================================

def create_market_scenarios() -> Dict:
    """Create 8 predefined market scenarios for analysis."""
    
    base_config = MarketConfig(
        demand_intercept=1000.0, demand_slope=-10.0, demand_shock_std=50.0,
        price_cap=1000.0, price_floor=0.0, min_bid=0.0, max_bid=200.0,
        bid_step=5.0, transparent=True
    )
    
    scenarios = {
        'sym_duopoly': {
            'name': 'Symmetric Duopoly',
            'description': 'Two identical generators',
            'n_generators': 2,
            'capacities': [400.0, 400.0],
            'marginal_costs': [30.0, 30.0],
            'market_config': base_config
        },
        'asym_duopoly_cap': {
            'name': 'Asymmetric Duopoly (Capacity)',
            'description': 'One large (60%), one small (40%)',
            'n_generators': 2,
            'capacities': [480.0, 320.0],
            'marginal_costs': [30.0, 30.0],
            'market_config': base_config
        },
        'asym_duopoly_cost': {
            'name': 'Asymmetric Duopoly (Cost)',
            'description': 'Different marginal costs',
            'n_generators': 2,
            'capacities': [400.0, 400.0],
            'marginal_costs': [20.0, 40.0],
            'market_config': base_config
        },
        'sym_triopoly': {
            'name': 'Symmetric Triopoly',
            'description': 'Three identical generators',
            'n_generators': 3,
            'capacities': [300.0, 300.0, 300.0],
            'marginal_costs': [30.0, 30.0, 30.0],
            'market_config': base_config
        },
        'oligopoly_5': {
            'name': 'Oligopoly (5 firms)',
            'description': 'Five generators with varying capacities',
            'n_generators': 5,
            'capacities': [250.0, 200.0, 180.0, 180.0, 190.0],
            'marginal_costs': [25.0, 30.0, 30.0, 35.0, 35.0],
            'market_config': base_config
        },
        'dominant_firm': {
            'name': 'Dominant Firm',
            'description': 'One large (50%) + competitive fringe',
            'n_generators': 4,
            'capacities': [500.0, 167.0, 167.0, 166.0],
            'marginal_costs': [25.0, 35.0, 35.0, 35.0],
            'market_config': base_config
        },
        'high_concentration': {
            'name': 'High Concentration',
            'description': 'Two large firms dominate (80%)',
            'n_generators': 4,
            'capacities': [400.0, 400.0, 100.0, 100.0],
            'marginal_costs': [30.0, 30.0, 40.0, 40.0],
            'market_config': base_config
        },
        'fragmented': {
            'name': 'Fragmented Market',
            'description': 'Many small generators',
            'n_generators': 8,
            'capacities': [125.0] * 8,
            'marginal_costs': [28.0, 30.0, 30.0, 32.0, 32.0, 35.0, 35.0, 38.0],
            'market_config': base_config
        }
    }
    
    return scenarios


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def run_comparative_analysis(scenarios: Dict = None, 
                            n_episodes: int = 10000,
                            n_runs: int = 3) -> pd.DataFrame:
    """
    Run comparative analysis across all scenarios.
    
    Parameters:
    -----------
    scenarios : Dict
        Dictionary of scenarios (if None, uses default)
    n_episodes : int
        Episodes per run
    n_runs : int
        Number of runs per scenario (for robustness)
        
    Returns:
    --------
    results_df : pd.DataFrame
        Comparative results across scenarios
    """
    if scenarios is None:
        scenarios = create_market_scenarios()
    
    print("="*70)
    print("COMPARATIVE ANALYSIS ACROSS MARKET STRUCTURES")
    print("="*70)
    
    all_results = []
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Generators: {scenario['n_generators']}")
        print(f"{'='*70}")
        
        scenario_runs = []
        
        for run in range(n_runs):
            print(f"\n  Run {run + 1}/{n_runs}...")
            
            # Create generators
            generator_configs = [
                GeneratorConfig(
                    id=i,
                    capacity=scenario['capacities'][i],
                    marginal_cost=scenario['marginal_costs'][i],
                    learning_rate=0.1,
                    epsilon=0.1
                )
                for i in range(scenario['n_generators'])
            ]
            
            # Run simulation
            market, analyzer = run_simulation(
                market_config=scenario['market_config'],
                generator_configs=generator_configs,
                n_episodes=n_episodes,
                eval_interval=n_episodes // 2
            )
            
            # Compute metrics
            window = min(1000, n_episodes // 2)
            metrics = compute_all_metrics(market, analyzer, window, scenario)
            scenario_runs.append(metrics)
        
        # Average across runs
        avg_metrics = {
            k: np.mean([r[k] for r in scenario_runs]) 
            for k in scenario_runs[0].keys()
        }
        
        result = {
            'scenario': scenario['name'],
            'scenario_key': scenario_key,
            'n_generators': scenario['n_generators'],
            **avg_metrics
        }
        
        all_results.append(result)
        
        print(f"\n  ✓ Completed {scenario['name']}")
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    return results_df


def compute_all_metrics(market: ElectricityMarket, analyzer: MarketAnalyzer,
                       window: int, scenario: Dict) -> Dict:
    """Compute comprehensive metrics for a scenario."""
    
    # Price metrics
    avg_price = np.mean(market.history['prices'][-window:])
    price_vol = np.std(market.history['prices'][-window:])
    
    # Competitive benchmark
    competitive_price = np.min(scenario['marginal_costs'])
    price_markup = (avg_price - competitive_price) / competitive_price * 100
    
    # Market structure
    hhi = analyzer.compute_hhi(window)
    lerner, _ = analyzer.compute_lerner_index(window)
    pcm = analyzer.compute_price_cost_margin(window)
    
    # Welfare
    avg_cs = np.mean(market.history['consumer_surplus'][-window:])
    avg_ps = np.mean(market.history['producer_surplus'][-window:])
    avg_ts = np.mean(market.history['total_surplus'][-window:])
    
    # Welfare loss
    demand_intercept = scenario['market_config'].demand_intercept
    demand_slope = scenario['market_config'].demand_slope
    competitive_quantity = demand_intercept + demand_slope * competitive_price
    p_max = (demand_intercept - 0) / abs(demand_slope)
    competitive_cs = 0.5 * (p_max - competitive_price) * competitive_quantity
    welfare_loss_pct = (competitive_cs - avg_cs) / competitive_cs * 100 if competitive_cs > 0 else 0
    
    # Nash equilibrium
    nash_test = analyzer.test_nash_equilibrium(n_tests=50)
    
    # Convergence
    convergence = analyzer.compute_convergence_metrics(window)
    
    return {
        'avg_price': avg_price,
        'price_volatility': price_vol,
        'competitive_price': competitive_price,
        'price_markup_pct': price_markup,
        'hhi': hhi,
        'lerner_index': lerner,
        'price_cost_margin': pcm,
        'consumer_surplus': avg_cs,
        'producer_surplus': avg_ps,
        'total_surplus': avg_ts,
        'welfare_loss_pct': welfare_loss_pct,
        'is_nash': 1 if nash_test['is_nash'] else 0,
        'max_nash_deviation': nash_test['max_deviation'],
        'policy_stability': np.mean(convergence['policy_stability'])
    }


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def analyze_demand_elasticity(base_scenario_key: str = 'sym_duopoly',
                              elasticities: List[float] = None,
                              n_episodes: int = 8000,
                              n_runs: int = 2) -> pd.DataFrame:
    """
    Analyze sensitivity to demand elasticity.
    
    Parameters:
    -----------
    base_scenario_key : str
        Base scenario to use
    elasticities : List[float]
        List of demand slopes to test (more negative = more elastic)
    n_episodes : int
        Episodes per run
    n_runs : int
        Runs per elasticity
        
    Returns:
    --------
    results_df : pd.DataFrame
        Sensitivity results
    """
    if elasticities is None:
        elasticities = [-5.0, -10.0, -15.0, -20.0, -30.0]
    
    scenarios = create_market_scenarios()
    base_scenario = scenarios[base_scenario_key]
    
    print(f"\nAnalyzing demand elasticity sensitivity for {base_scenario['name']}...")
    
    results = []
    
    for elasticity in elasticities:
        print(f"\n  Testing elasticity = {elasticity}")
        
        # Modify market config
        market_config = MarketConfig(
            demand_intercept=base_scenario['market_config'].demand_intercept,
            demand_slope=elasticity,
            demand_shock_std=base_scenario['market_config'].demand_shock_std,
            price_cap=base_scenario['market_config'].price_cap,
            price_floor=base_scenario['market_config'].price_floor,
            min_bid=base_scenario['market_config'].min_bid,
            max_bid=base_scenario['market_config'].max_bid,
            bid_step=base_scenario['market_config'].bid_step,
            transparent=base_scenario['market_config'].transparent
        )
        
        run_results = []
        
        for run in range(n_runs):
            generator_configs = [
                GeneratorConfig(
                    id=i,
                    capacity=base_scenario['capacities'][i],
                    marginal_cost=base_scenario['marginal_costs'][i],
                    learning_rate=0.1,
                    epsilon=0.1
                )
                for i in range(base_scenario['n_generators'])
            ]
            
            market, analyzer = run_simulation(
                market_config=market_config,
                generator_configs=generator_configs,
                n_episodes=n_episodes,
                eval_interval=n_episodes
            )
            
            window = min(1000, n_episodes // 2)
            avg_price = np.mean(market.history['prices'][-window:])
            lerner, _ = analyzer.compute_lerner_index(window)
            avg_cs = np.mean(market.history['consumer_surplus'][-window:])
            avg_ps = np.mean(market.history['producer_surplus'][-window:])
            
            run_results.append({
                'price': avg_price,
                'lerner': lerner,
                'cs': avg_cs,
                'ps': avg_ps
            })
        
        # Average across runs
        avg_results = {k: np.mean([r[k] for r in run_results]) for k in run_results[0].keys()}
        
        results.append({
            'elasticity': elasticity,
            **avg_results
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n  Elasticity Analysis Complete!")
    print(results_df.to_string(index=False))
    
    return results_df


def analyze_price_caps(base_scenario_key: str = 'sym_duopoly',
                       price_caps: List[float] = None,
                       n_episodes: int = 8000,
                       n_runs: int = 2) -> pd.DataFrame:
    """
    Analyze sensitivity to price caps.
    
    Parameters:
    -----------
    base_scenario_key : str
        Base scenario to use
    price_caps : List[float]
        List of price caps to test
    n_episodes : int
        Episodes per run
    n_runs : int
        Runs per cap
        
    Returns:
    --------
    results_df : pd.DataFrame
        Sensitivity results
    """
    if price_caps is None:
        price_caps = [60.0, 80.0, 100.0, 150.0, 300.0, 500.0]
    
    scenarios = create_market_scenarios()
    base_scenario = scenarios[base_scenario_key]
    
    print(f"\nAnalyzing price cap sensitivity for {base_scenario['name']}...")
    
    results = []
    
    for cap in price_caps:
        print(f"\n  Testing price cap = ${cap}")
        
        # Modify market config
        market_config = MarketConfig(
            demand_intercept=base_scenario['market_config'].demand_intercept,
            demand_slope=base_scenario['market_config'].demand_slope,
            demand_shock_std=base_scenario['market_config'].demand_shock_std,
            price_cap=cap,
            price_floor=base_scenario['market_config'].price_floor,
            min_bid=base_scenario['market_config'].min_bid,
            max_bid=base_scenario['market_config'].max_bid,
            bid_step=base_scenario['market_config'].bid_step,
            transparent=base_scenario['market_config'].transparent
        )
        
        run_results = []
        
        for run in range(n_runs):
            generator_configs = [
                GeneratorConfig(
                    id=i,
                    capacity=base_scenario['capacities'][i],
                    marginal_cost=base_scenario['marginal_costs'][i],
                    learning_rate=0.1,
                    epsilon=0.1
                )
                for i in range(base_scenario['n_generators'])
            ]
            
            market, analyzer = run_simulation(
                market_config=market_config,
                generator_configs=generator_configs,
                n_episodes=n_episodes,
                eval_interval=n_episodes
            )
            
            window = min(1000, n_episodes // 2)
            avg_price = np.mean(market.history['prices'][-window:])
            lerner, _ = analyzer.compute_lerner_index(window)
            
            # Binding frequency
            cap_binding = np.mean(np.array(market.history['prices'][-window:]) >= cap - 0.01)
            
            avg_cs = np.mean(market.history['consumer_surplus'][-window:])
            avg_ps = np.mean(market.history['producer_surplus'][-window:])
            
            run_results.append({
                'price': avg_price,
                'lerner': lerner,
                'cap_binding': cap_binding,
                'cs': avg_cs,
                'ps': avg_ps
            })
        
        avg_results = {k: np.mean([r[k] for r in run_results]) for k in run_results[0].keys()}
        
        results.append({
            'price_cap': cap,
            **avg_results
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n  Price Cap Analysis Complete!")
    print(results_df.to_string(index=False))
    
    return results_df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_scenario_test(scenario_key: str = 'sym_duopoly', 
                       n_episodes: int = 5000) -> None:
    """Quick test of a single scenario."""
    scenarios = create_market_scenarios()
    scenario = scenarios[scenario_key]
    
    print(f"\nQuick test: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    
    generator_configs = [
        GeneratorConfig(
            id=i,
            capacity=scenario['capacities'][i],
            marginal_cost=scenario['marginal_costs'][i]
        )
        for i in range(scenario['n_generators'])
    ]
    
    market, analyzer = run_simulation(
        market_config=scenario['market_config'],
        generator_configs=generator_configs,
        n_episodes=n_episodes,
        eval_interval=n_episodes
    )
    
    window = min(1000, n_episodes // 2)
    avg_price = np.mean(market.history['prices'][-window:])
    hhi = analyzer.compute_hhi(window)
    lerner, _ = analyzer.compute_lerner_index(window)
    
    print(f"\nResults:")
    print(f"  Average Price: ${avg_price:.2f}/MWh")
    print(f"  HHI: {hhi:.0f}")
    print(f"  Lerner Index: {lerner:.3f}")
    
    return market, analyzer
