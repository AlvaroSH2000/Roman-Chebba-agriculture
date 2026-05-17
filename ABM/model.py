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
        self.area_cell = self.voronoi["L_cell"] ** 2 / 1e3  # Convert from m^2 to ha

        with open("parameters.yaml", "r", encoding="utf-8") as f:
            raw_params = f.read()
        # Accept YAML-like files with trailing commas before comments or end-of-line.
        cleaned_params = re.sub(r",(?=\s*(#.*)?$)", "", raw_params, flags=re.MULTILINE)
        self.params = yaml.safe_load(cleaned_params)

        self.deltaT = 1
        self.lambda_v = self.params["lambda"]["v"]
        self.lambda_o = self.params["lambda"]["o"]
        self.lambda_w = self.params["lambda"]["w"]
        self.R_w = self.params.get("R_w", 0.5)
        self.sigma_w = 0
        self.epsilon = 1
        self.rain = "normal"
        self.temp = "normal"
        self.phi_base = self.params.get("food_requirement_base", 300)

        self.precipitation = self.get_precipitation()
        self.sites, self.num_agents = self.Initialize()
        self.villa_agents = list(self.sites)
        self.field_agents = [cell for villa in self.villa_agents for cell in villa.cells]
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Precipitation": "precipitation"},
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
        self.get_precipitation()
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
        
        return agents, num_agents
    
    def get_precipitation(self):
        """Generate precipitation values based on a modified gaussian distribution.
        """
        max_val = 600
        min_val = 350
        mu = (max_val + min_val) / 2
        sigma = 100
        p = 0.6827
        u = np.random.rand()
        if u <= p:
            # Part uniforme dins de l'interval central
            x = np.random.uniform(mu - sigma, mu + sigma)
            self.precipitation = x
            return x
        else:
            # Part gaussiana fora de l'interval (rejection sampling)
            while True:
                x = np.random.normal(mu, sigma)
                if abs(x - mu) > sigma:
                    self.precipitation = x
                    return x

    def rand(self, list):
        """Generate a random number between a and b.

        Args:
            list (list [a,b]): A list containing the lower and upper bounds
        Returns:
            float: A random number between a and b
        """
        return self.random.uniform(list[0], list[1])
