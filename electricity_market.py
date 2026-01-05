"""
Multi-Agent Reinforcement Learning for Electricity Markets
==========================================================

Complete implementation of Q-learning agents in day-ahead electricity markets.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for a single generator."""
    id: int
    capacity: float  # MW
    marginal_cost: float  # $/MWh
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1


@dataclass
class MarketConfig:
    """Configuration for the electricity market."""
    demand_intercept: float = 1000.0
    demand_slope: float = -10.0
    demand_shock_std: float = 50.0
    price_cap: float = 1000.0
    price_floor: float = 0.0
    min_bid: float = 0.0
    max_bid: float = 200.0
    bid_step: float = 5.0
    transparent: bool = True
    
    def get_bid_space(self) -> np.ndarray:
        """Return discretized bid space."""
        return np.arange(self.min_bid, self.max_bid + self.bid_step, self.bid_step)


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningGenerator:
    """Generator that learns bidding strategies using Q-learning."""
    
    def __init__(self, config: GeneratorConfig, market_config: MarketConfig):
        self.config = config
        self.market_config = market_config
        
        # Bid space
        self.bid_space = market_config.get_bid_space()
        self.n_actions = len(self.bid_space)
        
        # State discretization
        self.demand_states = self._discretize_demand_states()
        self.n_states = len(self.demand_states)
        
        # Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # History
        self.action_counts = np.zeros((self.n_states, self.n_actions))
        self.rewards_history = []
        self.bids_history = []
        self.profits_history = []
        
        # Epsilon decay
        self.epsilon = config.epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
    def _discretize_demand_states(self, n_states: int = 10) -> np.ndarray:
        """Create discrete demand states."""
        min_d = self.market_config.demand_intercept - 3 * self.market_config.demand_shock_std
        max_d = self.market_config.demand_intercept + 3 * self.market_config.demand_shock_std
        return np.linspace(min_d, max_d, n_states)
    
    def get_state_index(self, demand: float) -> int:
        """Map continuous demand to discrete state."""
        distances = np.abs(self.demand_states - demand)
        return np.argmin(distances)
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state, :])
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"""
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state, :])
        self.Q[state, action] = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.action_counts[state, action] += 1
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_bid(self, state: int, training: bool = True) -> float:
        """Get bid for current state."""
        action = self.choose_action(state, training)
        return self.bid_space[action]
    
    def get_policy(self) -> np.ndarray:
        """Get greedy policy."""
        return np.argmax(self.Q, axis=1)


# ============================================================================
# ELECTRICITY MARKET
# ============================================================================

class ElectricityMarket:
    """Day-ahead uniform price electricity market."""
    
    def __init__(self, generators: List[QLearningGenerator], market_config: MarketConfig):
        self.generators = generators
        self.config = market_config
        self.n_generators = len(generators)
        
        # History
        self.history = {
            'prices': [], 'quantities': [], 'generator_outputs': [],
            'generator_profits': [], 'bids': [], 'demands': [],
            'consumer_surplus': [], 'producer_surplus': [], 'total_surplus': []
        }
    
    def sample_demand(self) -> float:
        """Sample demand with random shock."""
        shock = np.random.normal(0, self.config.demand_shock_std)
        return max(0, self.config.demand_intercept + shock)
    
    def inverse_demand(self, quantity: float, demand_level: float) -> float:
        """Calculate price from quantity: P = (D - Q) / |slope|"""
        price = (demand_level - quantity) / abs(self.config.demand_slope)
        return np.clip(price, self.config.price_floor, self.config.price_cap)
    
    def clear_market(self, bids: np.ndarray, capacities: np.ndarray, 
                     demand_level: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Clear market using merit order dispatch."""
        order = np.argsort(bids)
        sorted_bids = bids[order]
        sorted_caps = capacities[order]
        cumulative_capacity = np.cumsum(sorted_caps)
        
        quantities = np.zeros(self.n_generators)
        best_price = self.config.price_floor
        best_quantities = quantities.copy()
        
        for i in range(self.n_generators):
            test_quantities = np.zeros(self.n_generators)
            
            if i > 0:
                test_quantities[order[:i]] = sorted_caps[:i]
            
            total_dispatched = cumulative_capacity[i-1] if i > 0 else 0
            quantity_demanded = demand_level + self.config.demand_slope * sorted_bids[i]
            
            if quantity_demanded > total_dispatched:
                remaining = min(sorted_caps[i], quantity_demanded - total_dispatched)
                test_quantities[order[i]] = max(0, remaining)
                total_dispatched += test_quantities[order[i]]
                
                test_price = self.inverse_demand(total_dispatched, demand_level)
                test_price = max(test_price, sorted_bids[i])
                
                if test_price >= sorted_bids[i]:
                    best_price = test_price
                    best_quantities = test_quantities.copy()
        
        if np.sum(best_quantities) == 0:
            best_price = self.config.price_floor
        
        clearing_price = min(best_price, self.config.price_cap)
        accepted = best_quantities > 0
        
        return clearing_price, best_quantities, accepted
    
    def calculate_welfare(self, price: float, total_quantity: float,
                         demand_level: float, quantities: np.ndarray,
                         costs: np.ndarray) -> Dict[str, float]:
        """Calculate welfare metrics."""
        p_max = self.inverse_demand(0, demand_level)
        consumer_surplus = 0.5 * (p_max - price) * total_quantity
        producer_surplus = sum((price - mc) * q for q, mc in zip(quantities, costs) if q > 0)
        total_surplus = consumer_surplus + producer_surplus
        
        competitive_price = np.min(costs)
        competitive_quantity = demand_level + self.config.demand_slope * competitive_price
        competitive_cs = 0.5 * (p_max - competitive_price) * competitive_quantity
        deadweight_loss = max(0, competitive_cs - total_surplus)
        
        return {
            'consumer_surplus': consumer_surplus,
            'producer_surplus': producer_surplus,
            'total_surplus': total_surplus,
            'deadweight_loss': deadweight_loss,
            'competitive_price': competitive_price,
            'competitive_quantity': competitive_quantity
        }
    
    def step(self, training: bool = True) -> Dict:
        """Execute one market period."""
        demand_level = self.sample_demand()
        state = self.generators[0].get_state_index(demand_level)
        
        bids = np.array([gen.get_bid(state, training) for gen in self.generators])
        capacities = np.array([gen.config.capacity for gen in self.generators])
        costs = np.array([gen.config.marginal_cost for gen in self.generators])
        
        price, quantities, accepted = self.clear_market(bids, capacities, demand_level)
        
        profits = np.array([
            (price - gen.config.marginal_cost) * quantities[i] if quantities[i] > 0 else 0
            for i, gen in enumerate(self.generators)
        ])
        
        welfare = self.calculate_welfare(price, np.sum(quantities), demand_level, quantities, costs)
        
        if training:
            next_demand = self.sample_demand()
            next_state = self.generators[0].get_state_index(next_demand)
            
            for i, gen in enumerate(self.generators):
                action = np.where(gen.bid_space == bids[i])[0][0]
                gen.update_q_value(state, action, profits[i], next_state)
                gen.rewards_history.append(profits[i])
                gen.bids_history.append(bids[i])
                gen.profits_history.append(profits[i])
                gen.decay_epsilon()
        
        # Store history
        self.history['prices'].append(price)
        self.history['quantities'].append(np.sum(quantities))
        self.history['generator_outputs'].append(quantities.copy())
        self.history['generator_profits'].append(profits.copy())
        self.history['bids'].append(bids.copy())
        self.history['demands'].append(demand_level)
        self.history['consumer_surplus'].append(welfare['consumer_surplus'])
        self.history['producer_surplus'].append(welfare['producer_surplus'])
        self.history['total_surplus'].append(welfare['total_surplus'])
        
        return {
            'price': price, 'quantities': quantities, 'profits': profits,
            'bids': bids, 'demand': demand_level, 'welfare': welfare
        }


# ============================================================================
# MARKET ANALYZER
# ============================================================================

class MarketAnalyzer:
    """Analyze market outcomes and compute metrics."""
    
    def __init__(self, market: ElectricityMarket):
        self.market = market
        self.generators = market.generators
    
    def compute_hhi(self, window: int = 1000) -> float:
        """Compute Herfindahl-Hirschman Index."""
        recent_outputs = self.market.history['generator_outputs'][-window:]
        avg_outputs = np.mean(recent_outputs, axis=0)
        total_output = np.sum(avg_outputs)
        if total_output == 0:
            return 0
        shares = avg_outputs / total_output
        return np.sum((shares * 100) ** 2)
    
    def compute_lerner_index(self, window: int = 1000) -> Tuple[float, np.ndarray]:
        """Compute Lerner Index: L = (P - MC) / P"""
        recent_prices = self.market.history['prices'][-window:]
        avg_price = np.mean(recent_prices)
        recent_outputs = self.market.history['generator_outputs'][-window:]
        avg_outputs = np.mean(recent_outputs, axis=0)
        costs = np.array([gen.config.marginal_cost for gen in self.generators])
        
        total_output = np.sum(avg_outputs)
        if total_output > 0:
            weighted_mc = np.sum(costs * avg_outputs) / total_output
            market_lerner = (avg_price - weighted_mc) / avg_price if avg_price > 0 else 0
        else:
            market_lerner = 0
        
        generator_lerner = np.array([
            (avg_price - costs[i]) / avg_price if avg_price > 0 else 0
            for i in range(len(self.generators))
        ])
        
        return market_lerner, generator_lerner
    
    def compute_price_cost_margin(self, window: int = 1000) -> float:
        """Compute price-cost margin (P-MC)/MC."""
        recent_prices = self.market.history['prices'][-window:]
        recent_outputs = self.market.history['generator_outputs'][-window:]
        avg_price = np.mean(recent_prices)
        avg_outputs = np.mean(recent_outputs, axis=0)
        costs = np.array([gen.config.marginal_cost for gen in self.generators])
        
        total_output = np.sum(avg_outputs)
        if total_output > 0:
            weighted_mc = np.sum(costs * avg_outputs) / total_output
            return (avg_price - weighted_mc) / weighted_mc if weighted_mc > 0 else 0
        return 0
    
    def test_nash_equilibrium(self, n_tests: int = 100) -> Dict:
        """Test if current policies form Nash equilibrium."""
        current_policies = [gen.get_policy() for gen in self.generators]
        nash_deviations = []
        
        for gen_idx in range(len(self.generators)):
            current_profits = []
            best_response_profits = []
            
            for _ in range(n_tests):
                demand = self.market.sample_demand()
                state = self.generators[0].get_state_index(demand)
                
                current_action = current_policies[gen_idx][state]
                bids = np.array([
                    self.generators[i].bid_space[current_policies[i][state]]
                    for i in range(len(self.generators))
                ])
                
                capacities = np.array([gen.config.capacity for gen in self.generators])
                price, quantities, _ = self.market.clear_market(bids, capacities, demand)
                
                current_profit = (price - self.generators[gen_idx].config.marginal_cost) * quantities[gen_idx]
                current_profits.append(current_profit)
                
                best_profit = current_profit
                for action in range(self.generators[gen_idx].n_actions):
                    test_bids = bids.copy()
                    test_bids[gen_idx] = self.generators[gen_idx].bid_space[action]
                    test_price, test_quantities, _ = self.market.clear_market(test_bids, capacities, demand)
                    test_profit = (test_price - self.generators[gen_idx].config.marginal_cost) * test_quantities[gen_idx]
                    best_profit = max(best_profit, test_profit)
                
                best_response_profits.append(best_profit)
            
            avg_deviation = np.mean(np.array(best_response_profits) - np.array(current_profits))
            nash_deviations.append(avg_deviation)
        
        return {
            'deviations': nash_deviations,
            'is_nash': all(d < 1.0 for d in nash_deviations),
            'max_deviation': max(nash_deviations)
        }
    
    def compute_convergence_metrics(self, window: int = 1000) -> Dict:
        """Compute convergence metrics."""
        recent_prices = self.market.history['prices'][-window:]
        price_volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
        
        recent_bids = self.market.history['bids'][-window:]
        bid_volatility = np.std(recent_bids, axis=0) / (np.mean(recent_bids, axis=0) + 1e-6)
        
        policy_changes = [np.mean(np.std(gen.Q, axis=1)) for gen in self.generators]
        
        return {
            'price_volatility': price_volatility,
            'bid_volatility': bid_volatility,
            'policy_stability': policy_changes,
            'price_trend': np.mean(recent_prices)
        }


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(market_config: MarketConfig,
                  generator_configs: List[GeneratorConfig],
                  n_episodes: int = 10000,
                  eval_interval: int = 1000) -> Tuple[ElectricityMarket, MarketAnalyzer]:
    """Run market simulation with Q-learning agents."""
    generators = [QLearningGenerator(config, market_config) for config in generator_configs]
    market = ElectricityMarket(generators, market_config)
    
    print(f"Starting simulation with {len(generators)} generators...")
    print(f"Episodes: {n_episodes}")
    print("-" * 60)
    
    for episode in range(n_episodes):
        market.step(training=True)
        
        if (episode + 1) % eval_interval == 0:
            analyzer = MarketAnalyzer(market)
            avg_price = np.mean(market.history['prices'][-eval_interval:])
            hhi = analyzer.compute_hhi(window=eval_interval)
            lerner, _ = analyzer.compute_lerner_index(window=eval_interval)
            convergence = analyzer.compute_convergence_metrics(window=eval_interval)
            
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Price: ${avg_price:.2f}/MWh")
            print(f"  HHI: {hhi:.0f}")
            print(f"  Lerner Index: {lerner:.3f}")
            print(f"  Price Volatility: {convergence['price_volatility']:.3f}")
            print(f"  Avg Epsilon: {np.mean([g.epsilon for g in generators]):.3f}")
            print("-" * 60)
    
    return market, MarketAnalyzer(market)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_learning_curves(market: ElectricityMarket, save_path: Optional[str] = None):
    """Plot learning curves and market outcomes."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    window = 100
    
    # Price evolution
    ax = axes[0, 0]
    prices_smooth = pd.Series(market.history['prices']).rolling(window).mean()
    ax.plot(prices_smooth, label='Market Price', linewidth=2)
    costs = [gen.config.marginal_cost for gen in market.generators]
    ax.axhline(np.min(costs), color='green', linestyle='--', label='Competitive', linewidth=2)
    ax.axhline(np.mean(market.history['prices'][-1000:]), color='red', linestyle='--', label='Converged', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('Price Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Generator profits
    ax = axes[0, 1]
    for i, gen in enumerate(market.generators):
        profits_smooth = pd.Series(gen.profits_history).rolling(window).mean()
        ax.plot(profits_smooth, label=f'Gen {i+1} (MC=${gen.config.marginal_cost})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Profit ($)')
    ax.set_title('Generator Profits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bid evolution
    ax = axes[1, 0]
    bids_array = np.array(market.history['bids'])
    for i in range(len(market.generators)):
        bids_smooth = pd.Series(bids_array[:, i]).rolling(window).mean()
        ax.plot(bids_smooth, label=f'Generator {i+1}', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Bid ($/MWh)')
    ax.set_title('Bid Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Welfare
    ax = axes[1, 1]
    cs_smooth = pd.Series(market.history['consumer_surplus']).rolling(window).mean()
    ps_smooth = pd.Series(market.history['producer_surplus']).rolling(window).mean()
    ts_smooth = pd.Series(market.history['total_surplus']).rolling(window).mean()
    ax.plot(cs_smooth, label='Consumer Surplus', linewidth=2)
    ax.plot(ps_smooth, label='Producer Surplus', linewidth=2)
    ax.plot(ts_smooth, label='Total Surplus', linewidth=2, linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Surplus ($)')
    ax.set_title('Welfare Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # HHI
    ax = axes[2, 0]
    hhi_history = []
    for i in range(100, len(market.history['generator_outputs']), 100):
        outputs = market.history['generator_outputs'][max(0, i-100):i]
        avg_outputs = np.mean(outputs, axis=0)
        total = np.sum(avg_outputs)
        if total > 0:
            shares = avg_outputs / total
            hhi = np.sum((shares * 100) ** 2)
            hhi_history.append(hhi)
    ax.plot(hhi_history, linewidth=2)
    ax.axhline(1800, color='orange', linestyle='--', label='Moderate')
    ax.axhline(2500, color='red', linestyle='--', label='High')
    ax.set_xlabel('Episode (x100)')
    ax.set_ylabel('HHI')
    ax.set_title('Market Concentration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Price distribution
    ax = axes[2, 1]
    recent_prices = market.history['prices'][-1000:]
    ax.hist(recent_prices, bins=30, alpha=0.7, edgecolor='black', density=True)
    kde = gaussian_kde(recent_prices)
    x_range = np.linspace(min(recent_prices), max(recent_prices), 100)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax.axvline(np.min(costs), color='green', linestyle='--', label=f'Competitive', linewidth=2)
    ax.set_xlabel('Price ($/MWh)')
    ax.set_ylabel('Density')
    ax.set_title('Price Distribution (Final 1000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_policy_analysis(market: ElectricityMarket, save_path: Optional[str] = None):
    """Plot learned policies."""
    n_gens = len(market.generators)
    fig, axes = plt.subplots(2, n_gens, figsize=(5*n_gens, 10))
    if n_gens == 1:
        axes = axes.reshape(-1, 1)
    
    for i, gen in enumerate(market.generators):
        # Q-value heatmap
        ax = axes[0, i]
        im = ax.imshow(gen.Q, aspect='auto', cmap='RdYlGn', origin='lower')
        ax.set_xlabel('Action (Bid Index)')
        ax.set_ylabel('State (Demand Level)')
        ax.set_title(f'Gen {i+1}: Q-Values (MC=${gen.config.marginal_cost})')
        plt.colorbar(im, ax=ax)
        policy = gen.get_policy()
        for state in range(gen.n_states):
            ax.plot(policy[state], state, 'bo', markersize=8)
        
        # Policy curve
        ax = axes[1, i]
        policy_bids = [gen.bid_space[a] for a in policy]
        ax.plot(gen.demand_states, policy_bids, 'b-', linewidth=2, marker='o')
        ax.axhline(gen.config.marginal_cost, color='green', linestyle='--', label='MC', linewidth=2)
        ax.set_xlabel('Demand Level (MW)')
        ax.set_ylabel('Bid ($/MWh)')
        ax.set_title(f'Gen {i+1}: Learned Policy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_summary_report(market: ElectricityMarket, analyzer: MarketAnalyzer,
                         save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create summary report."""
    window = 1000
    avg_price = np.mean(market.history['prices'][-window:])
    avg_quantity = np.mean(market.history['quantities'][-window:])
    hhi = analyzer.compute_hhi(window)
    lerner, gen_lerner = analyzer.compute_lerner_index(window)
    pcm = analyzer.compute_price_cost_margin(window)
    avg_cs = np.mean(market.history['consumer_surplus'][-window:])
    avg_ps = np.mean(market.history['producer_surplus'][-window:])
    avg_ts = np.mean(market.history['total_surplus'][-window:])
    convergence = analyzer.compute_convergence_metrics(window)
    nash_test = analyzer.test_nash_equilibrium(n_tests=100)
    
    costs = np.array([gen.config.marginal_cost for gen in market.generators])
    competitive_price = np.min(costs)
    markup = ((avg_price - competitive_price) / competitive_price * 100)
    
    summary_data = {
        'Metric': ['Avg Price ($/MWh)', 'Competitive Price ($/MWh)', 'Price Markup (%)',
                  'Avg Quantity (MW)', 'HHI', 'Lerner Index', 'Price-Cost Margin',
                  'Consumer Surplus ($)', 'Producer Surplus ($)', 'Total Surplus ($)',
                  'Price Volatility', 'Is Nash', 'Max Nash Deviation ($)'],
        'Value': [f'{avg_price:.2f}', f'{competitive_price:.2f}', f'{markup:.1f}%',
                 f'{avg_quantity:.1f}', f'{hhi:.0f}', f'{lerner:.3f}', f'{pcm:.3f}',
                 f'{avg_cs:.2f}', f'{avg_ps:.2f}', f'{avg_ts:.2f}',
                 f'{convergence["price_volatility"]:.3f}',
                 'Yes' if nash_test['is_nash'] else 'No',
                 f'{nash_test["max_deviation"]:.2f}']
    }
    summary_df = pd.DataFrame(summary_data)
    
    gen_data = []
    for i, gen in enumerate(market.generators):
        recent_outputs = np.array(market.history['generator_outputs'][-window:])
        avg_output = np.mean(recent_outputs[:, i])
        capacity_factor = avg_output / gen.config.capacity * 100
        avg_profit = np.mean(gen.profits_history[-window:])
        avg_bid = np.mean(gen.bids_history[-window:])
        gen_data.append({
            'Generator': i + 1, 'Capacity (MW)': gen.config.capacity,
            'Marginal Cost ($/MWh)': gen.config.marginal_cost,
            'Avg Bid ($/MWh)': f'{avg_bid:.2f}', 'Avg Output (MW)': f'{avg_output:.1f}',
            'Capacity Factor (%)': f'{capacity_factor:.1f}', 'Avg Profit ($)': f'{avg_profit:.2f}',
            'Lerner Index': f'{gen_lerner[i]:.3f}'
        })
    gen_df = pd.DataFrame(gen_data)
    
    print("\n" + "="*70)
    print("MARKET SUMMARY REPORT")
    print("="*70)
    print("\nAggregate Metrics:")
    print(summary_df.to_string(index=False))
    print("\nGenerator-Specific Metrics:")
    print(gen_df.to_string(index=False))
    print("="*70)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n" + "MARKET SUMMARY REPORT\n" + "="*70 + "\n\n")
            f.write("Aggregate Metrics:\n" + summary_df.to_string(index=False) + "\n\n")
            f.write("Generator-Specific Metrics:\n" + gen_df.to_string(index=False) + "\n")
            f.write("="*70 + "\n")
    
    return summary_df, gen_df
