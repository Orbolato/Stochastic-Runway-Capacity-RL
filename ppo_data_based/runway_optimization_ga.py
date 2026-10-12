import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import random
from deap import base, creator, tools, algorithms

class RunwayOptimizerGA:
    """
    Runway Configuration and Capacity Optimization using Genetic Algorithms.
    
    This class implements a meta-heuristic approach to the airport runway
    configuration problem, solving for the optimal runway configurations and service
    rates across a planning horizon to minimize total costs (queue lengths and configuration
    switches).
    """
    
    def __init__(
        self,
        alpha: float,
        delta: float,
        data: pd.DataFrame,
        u: np.ndarray,
        u_max: int,
        v: np.ndarray,
        v_max: int,
        possible_actions: List[List[int]],
        valid_bool: np.ndarray,
        wind_states: List[List[int]]
    ):
        """
        Initialize the runway optimizer with genetic algorithm approach.
        
        Args:
            alpha: Weight parameter for arrival queue in cost function
            delta: Weight parameter for configuration switches in cost function
            data: DataFrame containing weather and traffic data for optimization
            u: Arrival capacity matrix indexed by [weather_condition, runway_config]
            v: Departure capacity matrix indexed by [weather_condition, runway_config, arrival_rate]
            possible_actions: List of all possible actions as [RC, arr_rate, dep_rate]
            valid_bool: Boolean mask for valid actions by [wind_state, weather_cond, arr_queue, dep_queue]
            wind_states: Available runway configurations for each wind state
        """
        # Store parameters
        self.alpha = alpha
        self.delta = delta
        self.data = data
        self.u = u
        self.u_max = u_max
        self.v = v
        self.v_max = v_max
        self.possible_actions = np.array(possible_actions)
        self.valid_bool = valid_bool
        self.wind_states = wind_states
        
        # For storing results
        self.results = {}
        
        # GA parameters
        self.population_size = 100
        self.generations = 100
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        
        # Initialize DEAP classes only once
        if not hasattr(creator, "FitnessMin") or not isinstance(creator.FitnessMin, type):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual") or not isinstance(creator.Individual, type):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
    def optimize_day(self, day: int, verbose: bool = False, use_warm_start: bool = False) -> Dict[str, Any]:
        """
        Optimize runway operations for a single day using genetic algorithm.
        
        Args:
            day: Day index to optimize
            verbose: Whether to print optimization progress
            use_warm_start: Whether to use baseline solution as part of initial population
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract data for the specified day
        day_data = self.data.loc[self.data['day'] == day].reset_index(drop=True)
        time_steps = len(day_data)
        
        if verbose:
            print(f"Optimizing day {day} with {time_steps} time steps using Genetic Algorithm")
        
        # Get weather conditions and wind states for this day
        weather_conditions = day_data['weather_condition'].astype(int).tolist()
        wind_states = day_data['wind_state'].astype(int).tolist()
        arrival_demands = day_data['arrival_demand_40'].tolist()
        departure_demands = day_data['departure_demand'].tolist()
        
        # Define evaluation function
        def evaluate_individual(individual):
            # Parse individual
            configs = individual[:time_steps]
            arrival_rates = individual[time_steps:2*time_steps]
            
            # Validate configurations against wind constraints
            for t in range(time_steps):
                ws = wind_states[t]
                if configs[t] not in self.wind_states[ws]:
                    # If configuration is invalid for wind state, give a large penalty
                    return (1e9,)
            
            # Calculate service rates based on configurations and constraints
            arrivals_served = []
            departures_served = []
            
            for t in range(time_steps):
                wc = weather_conditions[t]
                rc = configs[t]
                
                # Get maximum arrival capacity for this config and weather
                max_arr_capacity = self.u[wc, rc][-1]
                
                # Determine arrival rate (capped by capacity and what's in queue)
                arr_rate = min(arrival_rates[t], max_arr_capacity)
                arrivals_served.append(arr_rate)
                
                # Determine departure rate based on arrival rate
                dep_capacity = self.v[wc, rc][arr_rate] if arr_rate in self.u[wc, rc] else 0
                departures_served.append(dep_capacity)
            
            # Calculate queue dynamics
            arrival_queue = [arrival_demands[0]]
            departure_queue = [departure_demands[0]]
            
            for t in range(time_steps):
                # Get current queue values
                current_arr_queue = arrival_queue[-1]
                current_dep_queue = departure_queue[-1]
                
                # Serve aircraft based on capacity and queue
                arr_served = min(arrivals_served[t], current_arr_queue)
                dep_served = min(departures_served[t], current_dep_queue)
                
                # Calculate next time step's queue if not at the end
                if t < time_steps - 1:
                    next_arr_queue = current_arr_queue - arr_served + arrival_demands[t+1]
                    next_dep_queue = current_dep_queue - dep_served + departure_demands[t+1]
                    
                    arrival_queue.append(next_arr_queue)
                    departure_queue.append(next_dep_queue)
            
            # Calculate configuration switches
            config_switches = [1]  # First period always counted as a switch
            for t in range(1, time_steps):
                config_switches.append(1 if configs[t] != configs[t-1] else 0)
            
            # Calculate objective function
            # Queue cost component
            queue_cost = sum(
                self.alpha * ((arrival_queue[t] - arr_served)**2) + 
                ((departure_queue[t] - dep_served)**2)
                for t, arr_served, dep_served in zip(range(time_steps), arrivals_served, departures_served)
            )
            
            # Switch cost component
            switch_cost = sum(
                config_switches[t] * self.delta * 
                (self.alpha * (arrival_queue[t]**2) + (departure_queue[t]**2))
                for t in range(time_steps)
            )
            
            total_cost = queue_cost + switch_cost
            
            return (total_cost,)
        
        # Setup genetic algorithm components
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Define individual structure: [configs, arrival_rates]
        # Gene range for configs: [0, 1, 2] (None, 10R_10L, 28L_28R)
        toolbox.register("config_gene", random.randint, 0, 2)
        # Gene range for arrival rates: [0, ..., u_max]
        toolbox.register("arrival_rate_gene", random.randint, 0, self.u_max)
        
        # Define individual creation
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.config_gene, toolbox.arrival_rate_gene), n=time_steps)
        
        # Define population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self._custom_mutate, indpb=0.1, time_steps=time_steps)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=self.population_size)
        
        # Include warm start solution in initial population if requested
        if use_warm_start:
            # Get baseline data for warm start
            base_configs = day_data['greedy_configuration'].tolist()
            
            # Calculate initial arrival rates based on greedy arrivals
            base_arrivals = day_data['greedy_arrivals_40'].tolist()
            
            # Create warm start individual
            warm_start_ind = creator.Individual(base_configs + base_arrivals)
            
            # Replace a random individual in the population with the warm start
            if len(pop) > 0:
                pop[0] = warm_start_ind
        
        # Run genetic algorithm
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the genetic algorithm
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob, 
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Extract best solution
        best_individual = hof[0]
        
        # Validate the best individual
        def is_valid_individual(individual, wind_states):
            configs = individual[:len(wind_states)]
            for t, (ws, rc) in enumerate(zip(wind_states, configs)):
                if rc not in self.wind_states[ws]:
                    return False
            return True
        
        # If best individual is invalid, find the best valid one in the population
        if not is_valid_individual(best_individual, wind_states):
            valid_individuals = [ind for ind in pop if is_valid_individual(ind, wind_states)]
            if valid_individuals:
                # Select the valid individual with the best fitness
                best_individual = min(valid_individuals, key=lambda ind: ind.fitness.values[0])
            else:
                raise ValueError("No valid individuals found in the population.")
                
        best_configs = best_individual[:time_steps]
        best_arrival_rates = best_individual[time_steps:2*time_steps]
        
        # Calculate derived values based on the best solution
        best_arrivals_served = []
        best_departures_served = []
        best_arrival_queue = [arrival_demands[0]]
        best_departure_queue = [departure_demands[0]]
        
        for t in range(time_steps):
            wc = weather_conditions[t]
            rc = best_configs[t]
            
            # Calculate arrival service rate
            arr_rate = min(best_arrival_rates[t], self.u[wc, rc][-1])
            arr_served = min(arr_rate, best_arrival_queue[-1])
            best_arrivals_served.append(arr_served)
            
            # Calculate departure service rate based on arrival rate
            dep_capacity = self.v[wc, rc][arr_rate] if arr_rate in self.u[wc, rc] else 0
            dep_served = min(dep_capacity, best_departure_queue[-1])
            best_departures_served.append(dep_served)
            
            # Calculate next period's queues if not at the end
            if t < time_steps - 1:
                next_arr_queue = best_arrival_queue[-1] - arr_served + arrival_demands[t+1]
                next_dep_queue = best_departure_queue[-1] - dep_served + departure_demands[t+1]
                
                best_arrival_queue.append(next_arr_queue)
                best_departure_queue.append(next_dep_queue)
        
        # Calculate configuration switches
        config_switches = [1]  # First period always counted as a switch
        for t in range(1, time_steps):
            config_switches.append(1 if best_configs[t] != best_configs[t-1] else 0)
        
        # Calculate objective function for the best solution
        queue_cost = sum(
            self.alpha * ((best_arrival_queue[t] - best_arrivals_served[t])**2) + 
            ((best_departure_queue[t] - best_departures_served[t])**2)
            for t in range(time_steps)
        )
        
        switch_cost = sum(
            config_switches[t] * self.delta * 
            (self.alpha * (best_arrival_queue[t]**2) + (best_departure_queue[t]**2))
            for t in range(time_steps)
        )
        
        total_cost = queue_cost + switch_cost
        
        # Compile solution
        solution = {
            'status': 'optimal',  # We assume GA found a good solution
            'termination_condition': 'maxGenerations',
            'objective': total_cost,
            'time_steps': time_steps,
            'configs': best_configs,
            'arrivals_served': best_arrivals_served,
            'departures_served': best_departures_served,
            'arrival_queue': best_arrival_queue,
            'departure_queue': best_departure_queue,
            'config_switches': config_switches,
            'weather_condition': weather_conditions,
            'wind_state': wind_states,
            'arrival_demand': arrival_demands,
            'departure_demand': departure_demands,
            'queue_cost': queue_cost,
            'switch_cost': switch_cost,
            'ga_logbook': logbook
        }
        
        # Store results
        self.results[day] = solution
        
        if verbose:
            print(f"Optimization completed with objective value: {total_cost}")
            print(f"Queue cost: {queue_cost}, Switch cost: {switch_cost}")
            print(f"Total configuration switches: {sum(config_switches)}")
        
        return solution
    
    def _custom_mutate(self, individual, indpb, time_steps):
        """
        Custom mutation operator that respects the structure of individuals.
        
        Args:
            individual: The individual to mutate
            indpb: Independent probability of each gene to be mutated
            time_steps: Number of time steps in the problem
            
        Returns:
            Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                if i < time_steps:  # Config gene
                    individual[i] = random.randint(0, 2)
                else:  # Arrival rate gene
                    individual[i] = random.randint(0, self.u_max)
        
        return (individual,)
    
    def solve_all_days(self, days: List[int], verbose: bool = False, use_warm_start: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Optimize all specified days using genetic algorithm.
        
        Args:
            days: List of days to optimize
            verbose: Whether to print solver output
            use_warm_start: Whether to use baseline solution in initial population
            
        Returns:
            Dictionary mapping day index to optimization results
        """
        results = {}
        for day in days:
            if verbose:
                print(f"\nOptimizing day {day} with GA...")
            
            results[day] = self.optimize_day(day, verbose, use_warm_start)
            
            if verbose:
                print(f"Day {day} completed with objective value: {results[day]['objective']}")
                
        return results
    
    def compare_to_baseline(self, day: int) -> Dict[str, Any]:
        """
        Compare optimization results to baseline for a specific day.
        
        Args:
            day: Day index to compare
            
        Returns:
            Dictionary with comparison metrics
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get optimization results
        opt_results = self.results[day]
        
        # Get baseline data
        day_data = self.data.loc[self.data['day'] == day].reset_index(drop=True)
        base_configs = day_data['greedy_configuration'].tolist()
        base_arrivals = day_data['greedy_arrivals_40'].tolist()
        base_departures = day_data['greedy_departures'].tolist()
        
        # Simulate baseline performance
        base_arrival_queue = [day_data.loc[0, 'arrival_demand_40']]
        base_departure_queue = [day_data.loc[0, 'departure_demand']]
        base_config_switches = []
        
        prev_config = 0  # Initial config
        
        for t in range(len(day_data)):
            # Check for config switch
            curr_config = base_configs[t]
            config_switch = int(curr_config != prev_config)
            base_config_switches.append(config_switch)
            prev_config = curr_config
            
            # Process arrivals and departures
            arr_served = base_arrivals[t]
            dep_served = base_departures[t]
            
            # Calculate new queues
            if t < len(day_data) - 1:
                new_arr_queue = base_arrival_queue[-1] - arr_served + day_data.loc[t+1, 'arrival_demand_40']
                new_dep_queue = base_departure_queue[-1] - dep_served + day_data.loc[t+1, 'departure_demand']
                base_arrival_queue.append(new_arr_queue)
                base_departure_queue.append(new_dep_queue)
        
        # Calculate base objective
        base_objective = sum(
            self.alpha * (base_arrival_queue[t] - base_arrivals[t])**2 + (base_departure_queue[t] - base_departures[t])**2 +
            base_config_switches[t] * self.delta * 
            (self.alpha * base_arrival_queue[t]**2 + base_departure_queue[t]**2)
            for t in range(len(day_data))
        )
        
        # Compare metrics
        comparison = {
            'day': day,
            'opt_objective': opt_results['objective'],
            'base_objective': base_objective,
            'improvement': (base_objective - opt_results['objective']) / abs(base_objective) * 100,
            'opt_configs': opt_results['configs'],
            'base_configs': base_configs,
            'opt_arr_queue': opt_results['arrival_queue'],
            'base_arr_queue': base_arrival_queue,
            'opt_dep_queue': opt_results['departure_queue'],
            'base_dep_queue': base_departure_queue,
            'opt_switches': sum(opt_results['config_switches']),
            'base_switches': sum(base_config_switches)
        }
        
        return comparison
    
    def visualize_results(self, day: int, figsize=(15, 10)):
        """
        Visualize optimization results for a specific day.
        
        Args:
            day: Day index to visualize
            figsize: Figure size for the plot
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get results
        comparison = self.compare_to_baseline(day)
        results = self.results[day]
        
        # Create figure
        fig, axs = plt.subplots(4, 1, figsize=figsize)
        
        # Time steps
        time_steps = range(len(comparison['opt_configs']))
        n_time_steps = len(time_steps)
        
        # Plot runway configurations
        axs[0].plot(time_steps, comparison['opt_configs'], 'b-', label='GA Optimized')
        axs[0].plot(time_steps, comparison['base_configs'], 'r--', label='Baseline')
        axs[0].set_ylabel('Runway Config')
        axs[0].set_title(f'Day {day} - Comparison (Improvement: {comparison["improvement"]:.2f}%)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot arrival queues - ensure we only plot the same number of points
        axs[1].plot(time_steps, comparison['opt_arr_queue'][:n_time_steps], 'b-', label='GA Optimized')
        axs[1].plot(time_steps, comparison['base_arr_queue'][:n_time_steps], 'r--', label='Baseline')
        axs[1].set_ylabel('Arrival Queue')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot departure queues - ensure we only plot the same number of points
        axs[2].plot(time_steps, comparison['opt_dep_queue'][:n_time_steps], 'b-', label='GA Optimized')
        axs[2].plot(time_steps, comparison['base_dep_queue'][:n_time_steps], 'r--', label='Baseline')
        axs[2].set_ylabel('Departure Queue')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot arrivals and departures served
        axs[3].plot(time_steps, results['arrivals_served'], 'b-', label='Arrivals Served')
        axs[3].plot(time_steps, results['departures_served'], 'g-', label='Departures Served')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Aircraft Served')
        axs[3].legend()
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"Day {day} Summary:")
        print(f"  Objective: GA Optimized = {comparison['opt_objective']:.2f}, Baseline = {comparison['base_objective']:.2f}")
        print(f"  Improvement: {comparison['improvement']:.2f}%")
        print(f"  Config Switches: GA Optimized = {comparison['opt_switches']}, Baseline = {comparison['base_switches']}")

    
    # def visualize_results(self, day: int, figsize=(15, 10)):
    #     """
    #     Visualize optimization results for a specific day.
        
    #     Args:
    #         day: Day index to visualize
    #         figsize: Figure size for the plot
    #     """
    #     if day not in self.results:
    #         raise ValueError(f"Day {day} has not been optimized yet")
        
    #     # Get results
    #     comparison = self.compare_to_baseline(day)
    #     results = self.results[day]
        
    #     # Create figure
    #     fig, axs = plt.subplots(4, 1, figsize=figsize)
        
    #     # Time steps
    #     time_steps = range(len(comparison['opt_configs']))
        
    #     # Plot runway configurations
    #     axs[0].plot(time_steps, comparison['opt_configs'], 'b-', label='GA Optimized')
    #     axs[0].plot(time_steps, comparison['base_configs'], 'r--', label='Baseline')
    #     axs[0].set_ylabel('Runway Config')
    #     axs[0].set_title(f'Day {day} - Comparison (Improvement: {comparison["improvement"]:.2f}%)')
    #     axs[0].legend()
    #     axs[0].grid(True)
        
    #     # Plot arrival queues
    #     axs[1].plot(time_steps, comparison['opt_arr_queue'][:-1], 'b-', label='GA Optimized')
    #     axs[1].plot(time_steps, comparison['base_arr_queue'][:-1], 'r--', label='Baseline')
    #     axs[1].set_ylabel('Arrival Queue')
    #     axs[1].legend()
    #     axs[1].grid(True)
        
    #     # Plot departure queues
    #     axs[2].plot(time_steps, comparison['opt_dep_queue'][:-1], 'b-', label='GA Optimized')
    #     axs[2].plot(time_steps, comparison['base_dep_queue'][:-1], 'r--', label='Baseline')
    #     axs[2].set_ylabel('Departure Queue')
    #     axs[2].legend()
    #     axs[2].grid(True)
        
    #     # Plot arrivals and departures served
    #     axs[3].plot(time_steps, results['arrivals_served'], 'b-', label='Arrivals Served')
    #     axs[3].plot(time_steps, results['departures_served'], 'g-', label='Departures Served')
    #     axs[3].set_xlabel('Time Step')
    #     axs[3].set_ylabel('Aircraft Served')
    #     axs[3].legend()
    #     axs[3].grid(True)
        
    #     plt.tight_layout()
    #     plt.show()
        
    #     # Print summary
    #     print(f"Day {day} Summary:")
    #     print(f"  Objective: GA Optimized = {comparison['opt_objective']:.2f}, Baseline = {comparison['base_objective']:.2f}")
    #     print(f"  Improvement: {comparison['improvement']:.2f}%")
    #     print(f"  Config Switches: GA Optimized = {comparison['opt_switches']}, Baseline = {comparison['base_switches']}")
        
    def plot_convergence(self, day: int, figsize=(10, 6)):
        """
        Plot the convergence of the genetic algorithm for a specific day.
        
        Args:
            day: Day index to plot
            figsize: Figure size for the plot
        """
        if day not in self.results or 'ga_logbook' not in self.results[day]:
            raise ValueError(f"Day {day} has not been optimized with GA yet")
        
        # Get logbook data
        logbook = self.results[day]['ga_logbook']
        
        # Extract statistics
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_avgs = logbook.select("avg")
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(gen, fit_mins, 'b-', label='Best Fitness')
        plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Cost)')
        plt.title(f'GA Convergence for Day {day}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale often better for visualizing convergence
        plt.tight_layout()
        plt.show()
    
        