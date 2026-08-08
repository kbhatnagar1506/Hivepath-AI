#!/usr/bin/env python3
"""
⚙️ Pareto Optimization Configuration for HivePath AI
===================================================
Configuration settings and parameters for Pareto frontier optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from enum import Enum

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    NSGA_II = "nsga_ii"
    SPEA2 = "spea2"
    MOEA_D = "moea_d"
    CUSTOM = "custom"

class SelectionMethod(Enum):
    """Selection method types"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    CROWDING = "crowding"

@dataclass
class ObjectiveConfig:
    """Configuration for an optimization objective"""
    name: str
    weight: float
    minimize: bool = True
    target_value: Optional[float] = None
    priority: int = 1
    bounds: tuple = (0.0, 1000.0)
    penalty_factor: float = 1.0

@dataclass
class VariableConfig:
    """Configuration for optimization variables"""
    name: str
    var_type: str  # 'continuous', 'discrete', 'integer'
    bounds: tuple
    default_value: float = 0.0
    step_size: Optional[float] = None

@dataclass
class ParetoOptimizationConfig:
    """Main configuration for Pareto optimization"""
    
    # Optimization parameters
    num_generations: int = 50
    population_size: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Strategy settings
    strategy: OptimizationStrategy = OptimizationStrategy.NSGA_II
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    
    # Objectives configuration
    objectives: List[ObjectiveConfig] = field(default_factory=lambda: [
        ObjectiveConfig("cost", 0.25, True, bounds=(0, 1000)),
        ObjectiveConfig("time", 0.20, True, bounds=(0, 500)),
        ObjectiveConfig("risk", 0.20, True, bounds=(0, 1)),
        ObjectiveConfig("accessibility", 0.20, False, bounds=(0, 100)),
        ObjectiveConfig("environmental", 0.15, True, bounds=(0, 100))
    ])
    
    # Variables configuration
    variables: List[VariableConfig] = field(default_factory=lambda: [
        VariableConfig("distance", "continuous", (10, 100)),
        VariableConfig("time", "continuous", (30, 300)),
        VariableConfig("vehicle_count", "integer", (1, 5)),
        VariableConfig("accessibility_score", "continuous", (40, 100)),
        VariableConfig("risk_score", "continuous", (0.1, 0.9))
    ])
    
    # Convergence criteria
    convergence_threshold: float = 0.001
    max_stagnation_generations: int = 10
    min_improvement: float = 0.01
    
    # Visualization settings
    enable_visualization: bool = True
    save_plots: bool = True
    plot_frequency: int = 5  # Plot every N generations
    
    # Output settings
    save_results: bool = True
    results_format: str = "json"  # json, csv, excel
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'optimization_parameters': {
                'num_generations': self.num_generations,
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            },
            'strategy_settings': {
                'strategy': self.strategy.value,
                'selection_method': self.selection_method.value,
                'tournament_size': self.tournament_size
            },
            'objectives': [
                {
                    'name': obj.name,
                    'weight': obj.weight,
                    'minimize': obj.minimize,
                    'target_value': obj.target_value,
                    'priority': obj.priority,
                    'bounds': obj.bounds,
                    'penalty_factor': obj.penalty_factor
                }
                for obj in self.objectives
            ],
            'variables': [
                {
                    'name': var.name,
                    'type': var.var_type,
                    'bounds': var.bounds,
                    'default_value': var.default_value,
                    'step_size': var.step_size
                }
                for var in self.variables
            ],
            'convergence_criteria': {
                'convergence_threshold': self.convergence_threshold,
                'max_stagnation_generations': self.max_stagnation_generations,
                'min_improvement': self.min_improvement
            },
            'visualization_settings': {
                'enable_visualization': self.enable_visualization,
                'save_plots': self.save_plots,
                'plot_frequency': self.plot_frequency
            },
            'output_settings': {
                'save_results': self.save_results,
                'results_format': self.results_format,
                'verbose': self.verbose
            }
        }
    
    def save_config(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"⚙️ Configuration saved to: {file_path}")
    
    @classmethod
    def load_config(cls, file_path: str) -> 'ParetoOptimizationConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load optimization parameters
        opt_params = data.get('optimization_parameters', {})
        config.num_generations = opt_params.get('num_generations', 50)
        config.population_size = opt_params.get('population_size', 100)
        config.elite_size = opt_params.get('elite_size', 10)
        config.mutation_rate = opt_params.get('mutation_rate', 0.1)
        config.crossover_rate = opt_params.get('crossover_rate', 0.8)
        
        # Load strategy settings
        strategy_settings = data.get('strategy_settings', {})
        config.strategy = OptimizationStrategy(strategy_settings.get('strategy', 'nsga_ii'))
        config.selection_method = SelectionMethod(strategy_settings.get('selection_method', 'tournament'))
        config.tournament_size = strategy_settings.get('tournament_size', 3)
        
        # Load objectives
        objectives_data = data.get('objectives', [])
        config.objectives = [
            ObjectiveConfig(
                name=obj['name'],
                weight=obj['weight'],
                minimize=obj['minimize'],
                target_value=obj.get('target_value'),
                priority=obj.get('priority', 1),
                bounds=tuple(obj.get('bounds', (0, 1000))),
                penalty_factor=obj.get('penalty_factor', 1.0)
            )
            for obj in objectives_data
        ]
        
        # Load variables
        variables_data = data.get('variables', [])
        config.variables = [
            VariableConfig(
                name=var['name'],
                var_type=var['type'],
                bounds=tuple(var.get('bounds', (0, 100))),
                default_value=var.get('default_value', 0.0),
                step_size=var.get('step_size')
            )
            for var in variables_data
        ]
        
        # Load convergence criteria
        conv_criteria = data.get('convergence_criteria', {})
        config.convergence_threshold = conv_criteria.get('convergence_threshold', 0.001)
        config.max_stagnation_generations = conv_criteria.get('max_stagnation_generations', 10)
        config.min_improvement = conv_criteria.get('min_improvement', 0.01)
        
        # Load visualization settings
        viz_settings = data.get('visualization_settings', {})
        config.enable_visualization = viz_settings.get('enable_visualization', True)
        config.save_plots = viz_settings.get('save_plots', True)
        config.plot_frequency = viz_settings.get('plot_frequency', 5)
        
        # Load output settings
        output_settings = data.get('output_settings', {})
        config.save_results = output_settings.get('save_results', True)
        config.results_format = output_settings.get('results_format', 'json')
        config.verbose = output_settings.get('verbose', True)
        
        return config

# Predefined configurations for different scenarios
class HivePathConfigs:
    """Predefined configurations for HivePath AI scenarios"""
    
    @staticmethod
    def get_standard_config() -> ParetoOptimizationConfig:
        """Standard configuration for general optimization"""
        return ParetoOptimizationConfig(
            num_generations=50,
            population_size=100,
            strategy=OptimizationStrategy.NSGA_II,
            selection_method=SelectionMethod.TOURNAMENT
        )
    
    @staticmethod
    def get_fast_config() -> ParetoOptimizationConfig:
        """Fast configuration for quick testing"""
        return ParetoOptimizationConfig(
            num_generations=20,
            population_size=50,
            strategy=OptimizationStrategy.NSGA_II,
            selection_method=SelectionMethod.TOURNAMENT,
            enable_visualization=False,
            verbose=False
        )
    
    @staticmethod
    def get_high_quality_config() -> ParetoOptimizationConfig:
        """High-quality configuration for production optimization"""
        return ParetoOptimizationConfig(
            num_generations=100,
            population_size=200,
            elite_size=20,
            strategy=OptimizationStrategy.NSGA_II,
            selection_method=SelectionMethod.CROWDING,
            convergence_threshold=0.0001,
            max_stagnation_generations=20
        )
    
    @staticmethod
    def get_accessibility_focused_config() -> ParetoOptimizationConfig:
        """Configuration focused on accessibility optimization"""
        config = ParetoOptimizationConfig()
        config.objectives = [
            ObjectiveConfig("accessibility", 0.40, False, bounds=(0, 100), priority=1),
            ObjectiveConfig("cost", 0.20, True, bounds=(0, 1000), priority=2),
            ObjectiveConfig("time", 0.20, True, bounds=(0, 500), priority=2),
            ObjectiveConfig("risk", 0.10, True, bounds=(0, 1), priority=3),
            ObjectiveConfig("environmental", 0.10, True, bounds=(0, 100), priority=3)
        ]
        return config
    
    @staticmethod
    def get_cost_focused_config() -> ParetoOptimizationConfig:
        """Configuration focused on cost optimization"""
        config = ParetoOptimizationConfig()
        config.objectives = [
            ObjectiveConfig("cost", 0.50, True, bounds=(0, 1000), priority=1),
            ObjectiveConfig("time", 0.25, True, bounds=(0, 500), priority=2),
            ObjectiveConfig("risk", 0.15, True, bounds=(0, 1), priority=3),
            ObjectiveConfig("accessibility", 0.05, False, bounds=(0, 100), priority=4),
            ObjectiveConfig("environmental", 0.05, True, bounds=(0, 100), priority=4)
        ]
        return config
    
    @staticmethod
    def get_environmental_focused_config() -> ParetoOptimizationConfig:
        """Configuration focused on environmental optimization"""
        config = ParetoOptimizationConfig()
        config.objectives = [
            ObjectiveConfig("environmental", 0.40, True, bounds=(0, 100), priority=1),
            ObjectiveConfig("cost", 0.20, True, bounds=(0, 1000), priority=2),
            ObjectiveConfig("time", 0.20, True, bounds=(0, 500), priority=2),
            ObjectiveConfig("accessibility", 0.10, False, bounds=(0, 100), priority=3),
            ObjectiveConfig("risk", 0.10, True, bounds=(0, 1), priority=3)
        ]
        return config

def main():
    """Main function to demonstrate configuration system"""
    
    print("⚙️ HivePath AI - Pareto Optimization Configuration")
    print("=" * 60)
    
    # Create standard configuration
    config = HivePathConfigs.get_standard_config()
    print("📋 Standard Configuration:")
    print(f"   Generations: {config.num_generations}")
    print(f"   Population Size: {config.population_size}")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   Objectives: {len(config.objectives)}")
    print(f"   Variables: {len(config.variables)}")
    
    # Save configuration
    config.save_config("pareto_config_standard.json")
    
    # Create accessibility-focused configuration
    acc_config = HivePathConfigs.get_accessibility_focused_config()
    print(f"\n♿ Accessibility-Focused Configuration:")
    print(f"   Primary Objective: {acc_config.objectives[0].name} (weight: {acc_config.objectives[0].weight})")
    acc_config.save_config("pareto_config_accessibility.json")
    
    # Create cost-focused configuration
    cost_config = HivePathConfigs.get_cost_focused_config()
    print(f"\n💰 Cost-Focused Configuration:")
    print(f"   Primary Objective: {cost_config.objectives[0].name} (weight: {cost_config.objectives[0].weight})")
    cost_config.save_config("pareto_config_cost.json")
    
    # Load and verify configuration
    loaded_config = ParetoOptimizationConfig.load_config("pareto_config_standard.json")
    print(f"\n✅ Configuration loaded successfully:")
    print(f"   Generations: {loaded_config.num_generations}")
    print(f"   Population Size: {loaded_config.population_size}")
    
    print(f"\n🎯 Available predefined configurations:")
    print("   - Standard: Balanced optimization")
    print("   - Fast: Quick testing")
    print("   - High Quality: Production optimization")
    print("   - Accessibility Focused: Prioritize accessibility")
    print("   - Cost Focused: Prioritize cost reduction")
    print("   - Environmental Focused: Prioritize sustainability")

if __name__ == "__main__":
    main()
