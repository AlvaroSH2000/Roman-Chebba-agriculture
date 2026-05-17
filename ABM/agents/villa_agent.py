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
    class Farmer:
        """
        A class representing a farmer with specific attributes.

        Attributes:
            gender (int): The gender of the farmer (0 for male, 1 for female).
            age (int): The current age of the farmer.
            death_age (int): The age at which the farmer is expected to die.
        """

        def __init__(self, gender, age, sons=0, patriarch=0):
            """
            Initialize a Farmer instance.

            Args:
                gender (int): The gender of the farmer.
                age (int): The current age of the farmer.
                sons (int): The number of sons the farmer has (default is 0).
                patriarch (int): 0 if the farmer is not a patriarch, 1 if they are (default is 0).
            """
            
            self.gender = gender
            self.age = age
            self.sons = sons
            self.patriarch = patriarch

    def __init__(self, model, ID, x, y, cells_indices, virtual_cells, site_type="villa"):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
            ID (int): The unique identifier for the agent
            cells_indices (list): The list of cell indices assigned to the agent
            virtual_cells (list): The list of virtual cells assigned to the agent
        """
        super().__init__(model)
        self.ID = ID
        self.x = x
        self.y = y
        self.site_type = site_type
        self.cells = self.initialize_cells(cells_indices)
        self.virtual_cells = virtual_cells

        self.workers = self.initialize_workers()
        self.phi_requirements = 1
        self.phi_ratio = 1

        self.assign_initial_areas()
        self.area, self.area_v, self.area_o, self.area_ow, self.area_w, self.area_s = self.get_areas()
        
        self.Q_v = 0
        self.Q_o = 0
        self.Q_w = 0

        self.consecutive_surplus = 0


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
            self.model.n_cells += 1
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
        def initial_wheat_proportion():
            self.update_food_requirements()
            initial_wheat_area = self.model.params["y_base"]["w"]*self.model.params["mu_site"]["villa"]["w"]/(self.phi_requirements)
            return initial_wheat_area/(self.model.area_cell*len(self.cells))
        for cell in self.cells:
            rand = self.random.random()
            if rand < 0.2:
                cell.crop = "functional"
            elif rand < 0.2+initial_wheat_proportion():
                cell.crop = "w"
            else:
                rand2 = self.random.random()
                if rand2 < self.model.params["alpha_v"]:
                    cell.crop = "v"
                else:
                    cell.crop = "o"
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
        
    def reassign_crops(self):
        """
        Reassingn crops based on the current food ratio
        """
        def ideal_wo_extra_cells():
            area_needed = self.model.params["y_base"]["w"]*self.model.params["mu_site"]["villa"]["w"]*self.params["beta_w"]/(self.phi_requirements-self.Q_w)
            return min(1,int(area_needed/(self.model.area_cell)))
        def ideal_w_cells():
            area_needed = self.model.params["y_base"]["w"]*self.model.params["mu_site"]["villa"]["w"]/(self.phi_requirements-self.Q_w)
            return min(1,int(area_needed/(self.model.area_cell)))

        if self.phi_ratio >= 1.2:
            self.consecutive_surplus += 1
            if self.consecutive_surplus < 3:
                return True
            for cell in self.cells:
                if cell.crop == "w":
                    rand = self.random.random()
                    if rand < self.model.params["alpha_v"]:
                        cell.update_crop("v")
                    else:
                        cell.update_crop("o")
                    return True
        else:
            cells_needed_wo = ideal_wo_extra_cells()
            
            cells_updated = 0
            for cell in self.cells:
                if cell.crop == "o":
                    cell.update_crop("ow")
                    cells_updated += 1
                    if cells_updated >= cells_needed_wo:
                        return True
            
            cells_needed_w = ideal_w_cells()
            cells_updated = 0
            for cell in self.cells:
                if cell.crop == "v":
                    cell.update_crop("w")
                    cells_updated += 1
                    if cells_updated >= cells_needed_w:
                        return True
            for cell in self.cells:
                if cell.crop == "o":
                    cell.update_crop("w")
                    cells_updated += 1
                    if cells_updated >= cells_needed_w:
                        return True
            for cell in self.cells:
                if cell.crop == "ow":
                    cell.update_crop("w")
                    cells_updated += 1
                    if cells_updated >= cells_needed_w:
                        return True


    def update_food_requirements(self):
        """
        Calculate the food requirements for the villa agent based on the number of workers and their individual food needs.
        """
        kappa = 0
        for n in range(self.workers):
            kappa += self.model.params['kappa_food']["adult_male"]
        self.phi_requirements = self.model.phi_base * kappa

    def update_food_ratio(self):
        """
        Update the food ratio for the villa agent based on the current food requirements and the available production of wheat.
        The food ratio is calculated as the ratio of available wheat to the total food requirements. If no requirements exist, the food ratio is set to 1.20.
        """
        if self.phi_requirements > 0:
            self.phi_ratio = self.Q_w / self.phi_requirements
        else:
            self.phi_ratio = 1.20
    
    def initialize_workers(self):
        """
        Initializes the workers for the villa agent. Each worker is represented as a Farmer instance with specific attributes

        """
        self.workers = []
        for n in range(15):
            self.workers.append(self.Farmer(gender=0, age=self.model.random.randint(16, 50)))
        return self.workers
    
    def population_step(self):
        """
        Update the population of the villa agent based on the population growth rate and the food ratio.
        """
        for n in range(self.workers):
            if self.model.get_mortality_by_age(self.workers[n].age, self.phi_ratio):
                self.workers.pop(n)
        if self.phi_ratio > 1.2:
            self.workers.append(self.Farmer(gender=0, age=self.model.random.randint(16, 50)))

    def step(self):
        self.update_production()
        self.update_food_requirements()
        self.update_food_ratio()
        self.area, self.area_v, self.area_o, self.area_ow, self.area_w, self.area_s = self.get_areas()
        self.reassign_crops()
        self.population_step()
