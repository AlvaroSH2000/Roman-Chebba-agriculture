"""
Chebba Farms Model
=====================

"""

from mesa import Model
from mesa.datacollection import DataCollector
from agents.villa_agent import VillaAgent
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import re
import ast
from scipy.stats import alpha, beta, rv_continuous


class ChebbaFarms(Model):
    """A simple model of an economy where agents exchange currency at random.

    All agents begin with one unit of currency, and each time step agents can give
    a unit of currency to another agent in the same cell. Over time, this produces
    a highly skewed distribution of wealth.

    Attributes:
        num_agents (int): Number of agents in the model
        positions (generator): Generator for agent positions
        grid (VoronoiGrid): The space in which agents move
        running (bool): Whether the model should continue running
        datacollector (DataCollector): Collects and stores model data
    """

    def __init__(self, seed=0):
        """Initialize the model.

        Args:
            seed (int, optional): Random seed. Defaults to 0.
        """
        super().__init__(seed=seed)

        self.sites_df = pd.read_csv("input_data/sites_voronoi.csv")
        self.voronoi = pd.read_pickle("input_data/regular_grid_with_voronoi.pkl")
        self.area_cell = self.voronoi["L_cell"] ** 2 / 1e4  # Convert from m^2 to ha

        with open("parameters.yaml", "r", encoding="utf-8") as f:
            raw_params = f.read()
        # Accept YAML-like files with trailing commas before comments or end-of-line.
        cleaned_params = re.sub(r",(?=\s*(#.*)?$)", "", raw_params, flags=re.MULTILINE)
        self.params = yaml.safe_load(cleaned_params)

        self.deltaT = 1
        self.n_step = 0
        self.n_cells = 0
        self.lambda_v = self.params["lambda"]["v"]
        self.lambda_o = self.params["lambda"]["o"]
        self.lambda_w = self.params["lambda"]["w"]
        self.R_w = self.params.get("R_w", 0.5)

        self.rain = "normal"
        self.temp = "normal"
        self.get_clime()
        self.phi_base = self.params.get("food_requirement_base", 300)

        self.sites, self.num_agents = self.Initialize()
        self.villa_agents = list(self.sites)
        self.field_agents = [cell for villa in self.villa_agents for cell in villa.cells]
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Rain": "rain", "Temperature": "temp"},
            agent_reporters={
                "ID": lambda a: getattr(a, "ID", np.nan),
                "AgentType": lambda a: type(a).__name__,
                "area_v": lambda a: getattr(a, "area_v", np.nan),
                "area_o": lambda a: getattr(a, "area_o", np.nan),
                "area_ow": lambda a: getattr(a, "area_ow", np.nan),
                "area_w": lambda a: getattr(a, "area_w", np.nan),
                "Q_v": lambda a: getattr(a, "Q_v", np.nan),
                "Q_o": lambda a: getattr(a, "Q_o", np.nan),
                "Q_w": lambda a: getattr(a, "Q_w", np.nan),
            },
        )


        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.n_step += 1
        self.get_clime()
        for agent in self.sites:
            agent.step()
        self.datacollector.collect(self)  # Collect data

    def Initialize(self):
        """Initialize the site agents and the field agents.

        """
        
        num_agents = self.sites_df.shape[0]
        agents = []

        for i in range(num_agents):
            cell_indices = self.sites_df.loc[i, "cell_indices"]
            if isinstance(cell_indices, str):
                cell_indices = ast.literal_eval(cell_indices)
            site_type = self.sites_df.loc[i, "type"] if "type" in self.sites_df.columns else "villa"
            agents.append(
                VillaAgent(
                    self,
                    self.sites_df.loc[i, "id"],
                    self.sites_df.loc[i, "x"],
                    self.sites_df.loc[i, "y"],
                    cell_indices,
                    [],
                    site_type,
                )
            )
        print(num_agents, "agents initialized.")
        print(self.n_cells, "field agents initialized.")
        return agents, num_agents
    
    def get_clime(self):
        """Generate precipitation and temperature values.
        """
        rand_t = self.random.random()
        if rand_t < self.params['temp_prob']['cool']:
            self.temp = "cool"
        elif rand_t < self.params['temp_prob']['cool']+self.params['temp_prob']['normal']:
            self.temp = "normal"
        elif rand_t < self.params['temp_prob']['cool']+self.params['temp_prob']['normal']+self.params['temp_prob']['warm']:
            self.temp = "warm"
        else:
            self.temp = "very_warm"
    
        
        rand_r = self.random.random()
        if rand_r < self.params['rain_prob']['very_dry']:
            self.rain = "very_dry"
        elif rand_r < self.params['rain_prob']['very_dry']+self.params['rain_prob']['dry']:
            self.rain = "dry"
        elif rand_r < self.params['rain_prob']['very_dry']+self.params['rain_prob']['dry']+self.params['rain_prob']['normal']:
            self.rain = "normal"
        else:
            self.rain = "humid"
        print(rand_t, rand_r)

    def get_mortality_by_age(self, age, food_ratio):
        """Get the mortality rate for a given age.

        Args:
            age (int): The age of the individual
        Returns:
            bool: True if the individual survies, False if it dies
        """
        def mu(x):
            return self.params["gompertz_makeham"]["alpha"] * np.exp(self.params["gompertz_makeham"]["beta"] * x) + self.params["gompertz_makeham"]["lamda"]
        def stress_multiplier(food_ratio, ages):
            if food_ratio < 0.6:
                return self.params["stress_multiplier_age"][0.6][ages]
            elif food_ratio < 0.8:
                return self.params["stress_multiplier_age"][0.8][ages]
            elif food_ratio < 1.0:
                return self.params["stress_multiplier_age"][1.0][ages]
            elif food_ratio < 1.2:
                return self.params["stress_multiplier_age"][1.2][ages]
            else:
                return self.params["stress_multiplier_age"]["else"][ages]

        if age == 0:
            if self.random.random() < self.rand(self.params["mortality"]["newborn"]):
                return False
        elif age < 4:
            if self.random.random() < min(self.params["q_max"], self.rand(self.params["mortality"]["infant"])*stress_multiplier(food_ratio, "child")):
                return False
        elif age < 9:
            if self.random.random() < min(self.params["q_max"], self.rand(self.params["mortality"]["child"])*stress_multiplier(food_ratio, "child")):
                return False
        elif age < 15:
            if self.random.random() < min(self.params["q_max"], self.rand(self.params["mortality"]["adolescent"])*stress_multiplier(food_ratio, "adolescent")):
                return False
        else:
            if self.random.random() < min(self.params["q_max"], mu(age)*stress_multiplier(food_ratio, "adult")):
                return False
        return True


    def rand(self, list):
        """Generate a random number between a and b.

        Args:
            list (list [a,b]): A list containing the lower and upper bounds
        Returns:
            float: A random number between a and b
        """
        return self.random.uniform(list[0], list[1])
