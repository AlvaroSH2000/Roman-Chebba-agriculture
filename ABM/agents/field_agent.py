from mesa import Agent
import numpy as np
import pandas as pd
from scipy.stats import gompertz

class FieldAgent(Agent):
    """
    This class represents a field agent in the Chebba Farms model.

    Attributes:
        wealth (int): The agent's current wealth (starts at 1)
    """

    def __init__(self, model, ID, site, area, soil, hydro):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
            ID (int): The unique identifier for the agent
            site (int): The site agent is located on
            area (float): The maximum area of the farm
            soil (int): The type of soil
            hydro (int): The type of hydrography
        """
        super().__init__(model)
        self.ID = ID
        self.site = site
        self.hydro = hydro
        self.area = area
        self.soil = soil
        self.hydro = hydro
        self.crop = None
        self.y_v = 0
        self.y_o = 0
        self.y_w = 0
        self.Q_v = 0
        self.Q_o = 0
        self.Q_w = 0
        self.beta = 0
        
        
    def update_yield(self):
        """
        Updates the yield values for the agent based on soil productivity, rain, temperature, and other factors.
    
        """
        def yield_function(crop):
            rain_state = getattr(self.model, "rain", "normal")
            temp_state = getattr(self.model, "temp", "normal")
            site_type = getattr(self.site, "site_type", "villa")
            if site_type not in self.model.params["mu_site"]:
                site_type = "villa"

            return (
                self.model.rand(self.model.params["y_base"][crop])
                * self.model.rand(self.model.params["gamma_soil"][int(self.soil)][crop])
                * self.model.params["gamma_hydro"][int(self.hydro)][crop]
                * self.model.params["eta_climate"][crop][temp_state][rain_state]
                * self.model.rand(self.model.params["mu_site"][site_type][crop])
            )
        if self.crop == "v":
            self.y_v = yield_function("v")
            self.y_o = 0
            self.y_w = 0
        elif self.crop == "o":
            self.y_v = 0
            self.y_o = yield_function("o")
            self.y_w = 0
        elif self.crop == "ow":
            self.y_v = 0
            self.y_o = yield_function("o")
            self.y_w = yield_function("w")
        elif self.crop == "w":
            self.y_v = 0
            self.y_o = 0
            self.y_w = yield_function("w")
        else:
            self.y_v = 0
            self.y_o = 0
            self.y_w = 0

    def update_production(self):
        """
        Updates the production values for wheat, wine, and oil based on the yield and area of the farm.
        The production is calculated as the product of the yield and the area for each crop type.
        """
        delta_t = getattr(self.model, "deltaT", 1)
        lambda_v = getattr(self.model, "lambda_v", self.model.params["lambda"]["v"])
        lambda_o = getattr(self.model, "lambda_o", self.model.params["lambda"]["o"])
        lambda_w = getattr(self.model, "lambda_w", self.model.params["lambda"]["w"])
        r_w = getattr(self.model, "R_w", self.model.params.get("R_w", 0.5))
        sigma_w = getattr(self.model, "sigma_w", self.model.params.get("sigma_w", 0))

        self.Q_v = self.area * self.y_v * delta_t * (1 - lambda_v)
        self.Q_o = self.area * self.y_o * delta_t * (1 - lambda_o) * self.model.random.choice(self.model.params["Alt_o"])
        self.Q_w = self.area * self.y_w * delta_t * self.beta * (1 - lambda_w) * r_w - sigma_w * self.area * self.beta

    def update_crop(self, new_crop):
        """
        Updates the crop type for the agent and recalculates the yield and production accordingly.

        Args:
            new_crop (str): The new crop type to be assigned to the agent. It can be "v", "o", "ow", or "w".
        """
        self.crop = new_crop
        if self.crop == "v":
            self.beta = 0
        elif self.crop == "o":
            self.beta = 0
        elif self.crop == "ow":
            self.beta = self.model.params['beta_w']
            print(self.beta)
        elif self.crop == "w":
            self.beta = 1


    def step(self):
        self.update_crop(self.crop)
        self.update_yield()
        self.update_production()