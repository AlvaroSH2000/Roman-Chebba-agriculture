from mesa import Agent
import numpy as np
import pandas as pd
from scipy.stats import gompertz
from agents.field_agent import FieldAgent

class VillaAgent(Agent):
    """
    This class represents a villa agent in the Chebba Farms model.

    Attributes:
        wealth (int): The agent's current wealth (starts at 1)
    """

    def __init__(self, model, ID, x, y, cells_indices, virtual_cells, site_type="villa"):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
            ID (int): The unique identifier for the agent
            cells (list): The list of cells assigned to the agent
            virtual_cells (list): The list of virtual cells assigned to the agent
        """
        super().__init__(model)
        self.ID = ID
        self.x = x
        self.y = y
        self.site_type = site_type
        self.cells = self.initialize_cells(cells_indices)
        self.virtual_cells = virtual_cells
        self.assign_initial_areas()
        self.area, self.area_v, self.area_o, self.area_ow, self.area_w, self.area_s = self.get_areas()
        
        self.Q_v = 0
        self.Q_o = 0
        self.Q_w = 0

        self.workers = 15
        self.phi_requirements = 1
        self.phi_factor = 1

    def initialize_cells(self, cell_indices):
        """
        Initializes the cells for the villa agent based on the provided cell indices.

        Args:
            cell_indices (list): A list of indices corresponding to the cells assigned to the agent.
        """
        cells = []
        ny = int(self.model.voronoi["ny"])
        nx = int(self.model.voronoi["nx"])
        soil_grid = np.asarray(self.model.voronoi["soil"])
        hydro_grid = np.asarray(self.model.voronoi["hydro"])

        for index in cell_indices:
            idx = int(index)
            row, col = np.unravel_index(idx, (ny, nx))
            soil = int(soil_grid[row, col])
            hydro = int(hydro_grid[row, col])
            cells.append(FieldAgent(self.model, idx, self, self.model.area_cell, soil, hydro))
        return cells

    def get_areas(self):
        """
        Calculate the total area, the area under vineyards, the area under olive groves,
         the area under olive groves mixed with wheat, the area unde pure wheat cultivation, and the funcitonal margin area.

        Returns:
            float: The total area of the agent in hectares (ha)
            flaot: The area under vineyards in hectares (ha)
            float: The area under olive groves in hectares (ha)
            float: The area under olive groves mixed with wheat in hectares (ha)
            float: The area under pure wheat cultivation in hectares (ha)
            float: The functional margin area in hectares (ha)
        """
        area_v = 0
        area_o = 0
        area_ow = 0
        area_w = 0
        area_s = 0

        for cell in self.cells:
            if cell.crop == "v":
                area_v += cell.area
            elif cell.crop == "o":
                area_o += cell.area
            elif cell.crop == "ow":
                area_ow += cell.area
            elif cell.crop == "w":
                area_w += cell.area
            elif cell.crop == "functional":
                area_s += cell.area
        area = area_v + area_o + area_ow + area_w + area_s

        return area, area_v, area_o, area_ow, area_w, area_s

    def assign_initial_areas(self):
        """
        Assigns crops to the agent's cells. 80% effective crops, which 40% (32% total) are vineyards, 30% (24% total) are olive groves, and
          30% (24% total) are olive groves mixed with wheat. The remaining 20% of the area is assigned as functional margin.
          No pure wheat cultivation is assigned.
        """
        for cell in self.cells:
            rand = self.random.random()
            if rand < 0.2:
                cell.crop = "v"
            elif rand < 0.4:
                cell.crop = "o"
            # elif rand < 0.6:
            #     cell.crop = "ow"
            elif rand < 0.8:
                cell.crop = "w"
            else:
                cell.crop = "functional"
            cell.update_crop(cell.crop)

    def update_production(self):
        """
        Updates the production values for wheat, wine, and oil based on the yield and area of the farm.
    
        """
        self.Q_v = 0
        self.Q_o = 0
        self.Q_w = 0

        for cell in self.cells:
            cell.step()
            self.Q_v += cell.Q_v
            self.Q_o += cell.Q_o
            self.Q_w += cell.Q_w
        
        
    def update_food_requirements(self):
        """
        Calculate the food requirements for the villa agent based on the number of workers and their individual food needs.
        """
        kappa = 0
        for n in range(self.workers):
            kappa += self.model.params['kappa_food']["adult_male"]
        self.phi_requirements = self.model.phi_base * kappa

    def update_food_factor(self):
        """
        Update the food factor for the villa agent based on the current food requirements and the available production of wheat.
        The food factor is calculated as the ratio of available wheat to the total food requirements. If no requirements exist, the food factor is set to 1.20.
        """
        if self.phi_requirements > 0:
            self.phi_factor = self.Q_w / self.phi_requirements
        else:
            self.phi_factor = 1.20
    
    def population_step(self):
        """
 
        """


    def step(self):
        self.update_production()
        self.area, self.area_v, self.area_o, self.area_ow, self.area_w, self.area_s = self.get_areas()
        self.update_food_requirements()
        self.update_food_factor()