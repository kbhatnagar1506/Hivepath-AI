#!/usr/bin/env python3
"""
🎯 Pareto Frontier Optimization for HivePath AI
===============================================
Multi-objective optimization using Pareto frontier analysis
Balances cost, time, risk, accessibility, and environmental impact
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.optimize import minimize
import json
from datetime import datetime

@dataclass
class OptimizationObjective:
    """Represents an optimization objective"""
    name: str
    weight: float
    minimize: bool = True
    target_value: float = None

@dataclass
class ParetoSolution:
    """Represents a Pareto optimal solution"""
    objectives: Dict[str, float]
    variables: Dict[str, float]
    dominance_rank: int = 0
    crowding_distance: float = 0.0

class ParetoFrontierOptimizer:
    """Pareto frontier optimization for multi-objective logistics"""
    
    def __init__(self):
        self.objectives = [
            OptimizationObjective("cost", 0.25, True),
            OptimizationObjective("time", 0.20, True),
            OptimizationObjective("risk", 0.20, True),
            OptimizationObjective("accessibility", 0.20, False),  # Maximize
            OptimizationObjective("environmental", 0.15, True)
        ]
        self.pareto_solutions = []
        self.dominated_solutions = []
        
    def calculate_objective_values(self, route_variables: Dict[str, float]) -> Dict[str, float]:
        """Calculate objective function values for given route variables"""
        
        # Extract variables
        distance = route_variables.get('distance', 0)
        time = route_variables.get('time', 0)
        vehicle_count = route_variables.get('vehicle_count', 1)
        accessibility_score = route_variables.get('accessibility_score', 50)
        risk_score = route_variables.get('risk_score', 0.5)
        
        # Cost objective (minimize)
        fuel_cost = distance * 0.12  # $0.12 per km
        driver_cost = time * 0.25    # $0.25 per minute
        vehicle_cost = vehicle_count * 50  # $50 per vehicle
        total_cost = fuel_cost + driver_cost + vehicle_cost
        
        # Time objective (minimize)
        total_time = time + (distance * 0.5)  # Base time + travel time
        
        # Risk objective (minimize)
        risk_penalty = risk_score * 100  # Risk penalty in minutes
        
        # Accessibility objective (maximize, so we minimize negative)
        accessibility_penalty = (100 - accessibility_score) * 0.1
        
        # Environmental objective (minimize carbon footprint)
        carbon_emissions = distance * 0.2 + vehicle_count * 5  # kg CO2
        
        return {
            "cost": total_cost,
            "time": total_time + risk_penalty,
            "risk": risk_penalty,
            "accessibility": -accessibility_penalty,  # Negative for minimization
            "environmental": carbon_emissions
        }
    
    def generate_random_solutions(self, num_solutions: int = 100) -> List[ParetoSolution]:
        """Generate random solutions for Pareto frontier analysis"""
        
        solutions = []
        
        for i in range(num_solutions):
            # Generate random route variables
            route_vars = {
                'distance': np.random.uniform(10, 100),  # km
                'time': np.random.uniform(30, 300),      # minutes
                'vehicle_count': np.random.randint(1, 6),
                'accessibility_score': np.random.uniform(40, 100),
                'risk_score': np.random.uniform(0.1, 0.9)
            }
            
            # Calculate objective values
            objectives = self.calculate_objective_values(route_vars)
            
            solution = ParetoSolution(
                objectives=objectives,
                variables=route_vars
            )
            solutions.append(solution)
        
        return solutions
    
    def dominates(self, solution1: ParetoSolution, solution2: ParetoSolution) -> bool:
        """Check if solution1 dominates solution2"""
        
        at_least_one_better = False
        
        for obj in self.objectives:
            obj_name = obj.name
            val1 = solution1.objectives[obj_name]
            val2 = solution2.objectives[obj_name]
            
            if obj.minimize:
                if val1 > val2:
                    return False  # solution1 is worse
                elif val1 < val2:
                    at_least_one_better = True
            else:
                if val1 < val2:
                    return False  # solution1 is worse
                elif val1 > val2:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def find_pareto_frontier(self, solutions: List[ParetoSolution]) -> Tuple[List[ParetoSolution], List[ParetoSolution]]:
        """Find Pareto optimal solutions using dominance ranking"""
        
        pareto_solutions = []
        dominated_solutions = []
        
        for i, solution1 in enumerate(solutions):
            is_dominated = False
            
            for j, solution2 in enumerate(solutions):
                if i != j and self.dominates(solution2, solution1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution1)
            else:
                dominated_solutions.append(solution1)
        
        return pareto_solutions, dominated_solutions
    
    def calculate_crowding_distance(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Calculate crowding distance for diversity preservation"""
        
        if len(solutions) <= 2:
            for solution in solutions:
                solution.crowding_distance = float('inf')
            return solutions
        
        # Initialize crowding distances
        for solution in solutions:
            solution.crowding_distance = 0.0
        
        # Calculate for each objective
        for obj in self.objectives:
            obj_name = obj.name
            
            # Sort solutions by this objective
            sorted_solutions = sorted(solutions, key=lambda x: x.objectives[obj_name])
            
            # Set boundary solutions to infinite distance
            sorted_solutions[0].crowding_distance = float('inf')
            sorted_solutions[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = sorted_solutions[-1].objectives[obj_name] - sorted_solutions[0].objectives[obj_name]
            
            if obj_range > 0:
                # Calculate crowding distance for intermediate solutions
                for i in range(1, len(sorted_solutions) - 1):
                    distance = (sorted_solutions[i + 1].objectives[obj_name] - 
                              sorted_solutions[i - 1].objectives[obj_name]) / obj_range
                    sorted_solutions[i].crowding_distance += distance
        
        return solutions
    
    def nsga_ii_selection(self, population: List[ParetoSolution], num_selected: int) -> List[ParetoSolution]:
        """NSGA-II selection for Pareto frontier optimization"""
        
        # Fast non-dominated sorting
        pareto_fronts = []
        remaining_solutions = population.copy()
        
        while remaining_solutions:
            current_front = []
            dominated_solutions = []
            
            for solution in remaining_solutions:
                is_dominated = False
                for other in remaining_solutions:
                    if solution != other and self.dominates(other, solution):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(solution)
                else:
                    dominated_solutions.append(solution)
            
            pareto_fronts.append(current_front)
            remaining_solutions = dominated_solutions
        
        # Select solutions from fronts
        selected = []
        for front in pareto_fronts:
            if len(selected) + len(front) <= num_selected:
                selected.extend(front)
            else:
                # Calculate crowding distance for this front
                front = self.calculate_crowding_distance(front)
                # Sort by crowding distance (descending)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                # Add remaining solutions needed
                remaining_needed = num_selected - len(selected)
                selected.extend(front[:remaining_needed])
                break
        
        return selected
    
    def optimize_pareto_frontier(self, num_generations: int = 50, population_size: int = 100) -> Dict[str, Any]:
        """Main Pareto frontier optimization algorithm"""
        
        print("🎯 Starting Pareto Frontier Optimization")
        print("=" * 50)
        
        # Initialize population
        population = self.generate_random_solutions(population_size)
        
        pareto_front_history = []
        
        for generation in range(num_generations):
            print(f"🔄 Generation {generation + 1}/{num_generations}")
            
            # Find Pareto frontier
            pareto_solutions, dominated_solutions = self.find_pareto_frontier(population)
            
            # Calculate crowding distance
            pareto_solutions = self.calculate_crowding_distance(pareto_solutions)
            
            # Store Pareto front for this generation
            pareto_front_history.append({
                'generation': generation + 1,
                'pareto_size': len(pareto_solutions),
                'solutions': [
                    {
                        'objectives': sol.objectives,
                        'variables': sol.variables,
                        'crowding_distance': sol.crowding_distance
                    }
                    for sol in pareto_solutions
                ]
            })
            
            print(f"   📊 Pareto solutions: {len(pareto_solutions)}")
            print(f"   📊 Dominated solutions: {len(dominated_solutions)}")
            
            # NSGA-II selection for next generation
            if generation < num_generations - 1:
                population = self.nsga_ii_selection(population, population_size)
                
                # Add some random mutations for diversity
                for solution in population:
                    if np.random.random() < 0.1:  # 10% mutation rate
                        # Mutate variables slightly
                        for var_name in solution.variables:
                            if var_name in ['distance', 'time', 'accessibility_score', 'risk_score']:
                                mutation = np.random.normal(0, 0.1)
                                solution.variables[var_name] *= (1 + mutation)
                                solution.variables[var_name] = max(0, solution.variables[var_name])
                        
                        # Recalculate objectives
                        solution.objectives = self.calculate_objective_values(solution.variables)
        
        # Final Pareto frontier
        final_pareto, _ = self.find_pareto_frontier(population)
        final_pareto = self.calculate_crowding_distance(final_pareto)
        
        # Sort by crowding distance for best solutions
        final_pareto.sort(key=lambda x: x.crowding_distance, reverse=True)
        
        print(f"\n✅ Optimization Complete!")
        print(f"📊 Final Pareto frontier size: {len(final_pareto)}")
        
        return {
            'pareto_frontier': [
                {
                    'objectives': sol.objectives,
                    'variables': sol.variables,
                    'crowding_distance': sol.crowding_distance
                }
                for sol in final_pareto
            ],
            'generation_history': pareto_front_history,
            'optimization_parameters': {
                'num_generations': num_generations,
                'population_size': population_size,
                'objectives': [obj.name for obj in self.objectives]
            }
        }
    
    def visualize_pareto_frontier(self, results: Dict[str, Any], save_path: str = None):
        """Visualize Pareto frontier in 2D projections"""
        
        pareto_solutions = results['pareto_frontier']
        
        if len(pareto_solutions) < 2:
            print("⚠️ Not enough Pareto solutions for visualization")
            return
        
        # Create subplots for different objective pairs
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 HivePath AI Pareto Frontier Analysis', fontsize=16, fontweight='bold')
        
        objective_pairs = [
            ('cost', 'time'),
            ('cost', 'risk'),
            ('cost', 'accessibility'),
            ('time', 'risk'),
            ('time', 'environmental'),
            ('risk', 'environmental')
        ]
        
        for i, (obj1, obj2) in enumerate(objective_pairs):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Extract objective values
            x_values = [sol['objectives'][obj1] for sol in pareto_solutions]
            y_values = [sol['objectives'][obj2] for sol in pareto_solutions]
            
            # Plot Pareto frontier
            ax.scatter(x_values, y_values, c='red', s=50, alpha=0.7, label='Pareto Optimal')
            ax.scatter(x_values, y_values, c='red', s=20, alpha=1.0)
            
            # Connect points to show frontier
            sorted_indices = sorted(range(len(x_values)), key=lambda i: x_values[i])
            sorted_x = [x_values[i] for i in sorted_indices]
            sorted_y = [y_values[i] for i in sorted_indices]
            ax.plot(sorted_x, sorted_y, 'r--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel(f'{obj1.title()} ({"Minimize" if obj1 != "accessibility" else "Maximize"})')
            ax.set_ylabel(f'{obj2.title()} ({"Minimize" if obj2 != "accessibility" else "Maximize"})')
            ax.set_title(f'{obj1.title()} vs {obj2.title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Pareto frontier visualization saved to: {save_path}")
        
        plt.show()
    
    def analyze_pareto_solutions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pareto frontier solutions"""
        
        pareto_solutions = results['pareto_frontier']
        
        if not pareto_solutions:
            return {'error': 'No Pareto solutions found'}
        
        analysis = {
            'summary': {
                'total_solutions': len(pareto_solutions),
                'objectives_analyzed': len(self.objectives)
            },
            'objective_statistics': {},
            'best_solutions': {},
            'trade_off_analysis': {}
        }
        
        # Calculate statistics for each objective
        for obj in self.objectives:
            obj_name = obj.name
            values = [sol['objectives'][obj_name] for sol in pareto_solutions]
            
            analysis['objective_statistics'][obj_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'range': max(values) - min(values)
            }
        
        # Find best solutions for each objective
        for obj in self.objectives:
            obj_name = obj.name
            if obj.minimize:
                best_idx = min(range(len(pareto_solutions)), 
                             key=lambda i: pareto_solutions[i]['objectives'][obj_name])
            else:
                best_idx = max(range(len(pareto_solutions)), 
                             key=lambda i: pareto_solutions[i]['objectives'][obj_name])
            
            analysis['best_solutions'][obj_name] = {
                'solution_index': best_idx,
                'objectives': pareto_solutions[best_idx]['objectives'],
                'variables': pareto_solutions[best_idx]['variables']
            }
        
        # Trade-off analysis
        print("\n📊 Pareto Frontier Analysis")
        print("=" * 40)
        print(f"Total Pareto solutions: {len(pareto_solutions)}")
        
        for obj in self.objectives:
            obj_name = obj.name
            stats = analysis['objective_statistics'][obj_name]
            best_sol = analysis['best_solutions'][obj_name]
            
            print(f"\n🎯 {obj_name.title()} Objective:")
            print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"   Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"   Best value: {best_sol['objectives'][obj_name]:.2f}")
        
        return analysis

def main():
    """Main function to run Pareto frontier optimization"""
    
    print("🚀 HivePath AI - Pareto Frontier Optimization")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ParetoFrontierOptimizer()
    
    # Run optimization
    results = optimizer.optimize_pareto_frontier(
        num_generations=30,
        population_size=80
    )
    
    # Analyze results
    analysis = optimizer.analyze_pareto_solutions(results)
    
    # Visualize Pareto frontier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = f"pareto_frontier_visualization_{timestamp}.png"
    optimizer.visualize_pareto_frontier(results, viz_path)
    
    # Save results
    results_path = f"pareto_optimization_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis,
            'timestamp': timestamp
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"📊 Visualization saved to: {viz_path}")
    
    # Show top 5 Pareto solutions
    print(f"\n🏆 Top 5 Pareto Optimal Solutions:")
    print("-" * 50)
    
    for i, solution in enumerate(results['pareto_frontier'][:5]):
        print(f"\nSolution {i+1}:")
        print(f"  Objectives:")
        for obj_name, value in solution['objectives'].items():
            print(f"    {obj_name}: {value:.2f}")
        print(f"  Variables:")
        for var_name, value in solution['variables'].items():
            print(f"    {var_name}: {value:.2f}")
        print(f"  Crowding Distance: {solution['crowding_distance']:.3f}")

if __name__ == "__main__":
    main()
