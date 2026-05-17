from mesa import Agent
import numpy as np
import pandas as pd
from scipy.stats import gompertz

class FarmAgent(Agent):
    """
    This class represents a ville agent in the Chebba Farms model.

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

        def __init__(self, gender, age, death_age, sons=0, patriarch=0):
            """
            Initialize a Farmer instance.

            Args:
                gender (int): The gender of the farmer.
                age (int): The current age of the farmer.
                death_age (int): The age at which the farmer is expected to die.
                sons (int): The number of sons the farmer has (default is 0).
                patriarch (int): 0 if the farmer is not a patriarch, 1 if they are (default is 0).
            """
            
            self.gender = gender
            self.age = age
            self.death_age = death_age
            self.sons = sons
            self.patriarch = patriarch

    def __init__(self, model, ID, site_type, area, soil, dolia=0):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
            ID (int): The unique identifier for the agent
            site_type (str): The type of site ("Villa", "Farm")
            area (float): The maximum area of the farm
            soil (int): The type of soil
            dolia (int): The number of wine dolia (default is 0)
        """
        super().__init__(model)
        self.ID = ID
        self.site_type = site_type
        if self.site_type == "Farm":
            self.area = 3.25
        else:
            self.area = area
        self.maximum_extension = area
        self.soil = soil
        self.dolia = dolia
        self.wheat_prod, self.wine_prod, self.oil_prod = self.get_soil_productivity()
        self.opt_wheat, self.opt_wine, self.opt_oil = self.initialize_optimal_precipitation()
        self.wheat_factor, self.wine_factor, self.oil_factor = self.get_precipitation_factor()
        self.wheat, self.wine, self.oil = self.production_step()
        self.population = self.initialize_polulation()
        self.population_len = len(self.population)
        
        
    def get_soil_productivity(self):
        """
        Retrieves soil productivity data for the current farm agent based on its soil type.
        This method reads soil productivity values from an Excel file and assigns the 
        productivity values for wheat, wine, and oil to the respective attributes of the 
        farm agent. It then returns these productivity values.
        Returns:
            tuple: A tuple containing the productivity values for wheat, wine, and oil 
                   (wheat_prod, wine_prod, oil_prod).
    
        """
        
        df_soil = pd.read_excel("../input_files/Soil_productivity.xlsx", sheet_name="Hoja1")
        self.wheat_prod = df_soil[self.soil][0]
        self.wine_prod = df_soil[self.soil][1]
        self.oil_prod = df_soil[self.soil][2]
        return self.wheat_prod, self.wine_prod, self.oil_prod

    def initialize_optimal_precipitation(self):
        """
        Initializes the optimal precipitation ranges for different crops.

        This method sets the optimal precipitation ranges for wheat, wine, 
        and oil crops. The ranges are defined as lists containing the minimum 
        and maximum precipitation values (in millimeters/yr) required for optimal 
        growth of each crop.

        Returns:
            tuple: A tuple containing three lists:
                - opt_wheat (list): Optimal precipitation range for wheat [min, max].
                - opt_wine (list): Optimal precipitation range for wine [min, max].
                - opt_oil (list): Optimal precipitation range for oil [min, max].
        """
        self.opt_wheat = [350, 600]
        self.opt_wine = [500, 700]
        self.opt_oil = [350, 400]
        return self.opt_wheat, self.opt_wine, self.opt_oil
    
    def gaussian_modified_pdf(self, max_val, min_val, sigma=100, p=0.6827):
        """
        Computes a modified probability density function (PDF) that combines a uniform distribution 
        within a specified range and a Gaussian distribution outside that range.
        Parameters:
            x (float): The input value for which the PDF is evaluated.
            max_val (float): The maximum value of the uniform range.
            min_val (float): The minimum value of the uniform range.
        Returns:
            float: The computed value of the modified PDF at the given input `x`.
        Notes:
            - The uniform part of the distribution is defined between `mu - sigma` and `mu + sigma`, 
              where `mu` is the midpoint of `max_val` and `min_val`.
            - Outside the uniform range, the PDF transitions to a Gaussian distribution.
            - The Gaussian part is normalized to ensure continuity and proper scaling.
        """
        x = self.model.precipitation
        mu = (max_val + min_val) / 2
        if mu - sigma <= x <= mu + sigma:
            # Uniform part
            return 1
        else:
            # Gaussian part
            normalization = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((sigma) / sigma) ** 2)
            
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / normalization


    def get_precipitation_factor(self):
        """
        Calculate the precipitation factor for wheat, wine, and oil based on the optimal precipitation ranges
        with the precipitation of the model.
        """
        self.wheat_factor = self.gaussian_modified_pdf(self.opt_wheat[0], self.opt_wheat[1])
        self.wine_factor = self.gaussian_modified_pdf(self.opt_wine[0], self.opt_wine[1])
        self.oil_factor = self.gaussian_modified_pdf(self.opt_oil[0], self.opt_oil[1])
        return self.wheat_factor, self.wine_factor, self.oil_factor


    def production_step(self):
        """Calculate the production of wheat, wine, and oil based on the farm's area and soil type."""
        self.wheat_factor, self.wine_factor, self.oil_factor = self.get_precipitation_factor()
        if self.site_type == "Farm":
            self.wheat = 0.5*self.area * self.wheat_prod * self.wheat_factor
            self.wine = 0.25*self.area * self.wine_prod * self.wine_factor
            self.oil = 0.25*self.area * self.oil_prod * self.oil_factor
        else:
            if self.dolia == 0:
                self.oil = self.area * self.oil_prod * self.oil_factor
                self.wine = 0
                self.wheat = 0
            else:
                self.wine = self.area * self.wine_prod * self.wine_factor
                self.oil = 0
                self.wheat = 0
        return self.wheat, self.wine, self.oil

        
    def initialize_polulation(self):
        """
        Initializes the population of the farm agent with the family.

        Returns:
            list: A list containing the population.
        """
        if self.site_type == "Farm":
            initial_farmers = []
            initial_farmers.append(self.Farmer(gender=0, age=30, death_age=self.get_death_date(), patriarch=1))
            initial_farmers.append(self.Farmer(gender=1, age=27, death_age=self.get_death_date(), sons=3))
            initial_farmers.append(self.Farmer(gender=0, age=10, death_age=self.get_death_date()))
            initial_farmers.append(self.Farmer(gender=0, age=8, death_age=self.get_death_date()))
            initial_farmers.append(self.Farmer(gender=1, age=5, death_age=self.get_death_date()))
            return initial_farmers
        else:
            return []
        
    def get_death_date(self):
        """
        Calculates the death date of the farmer based gompertz distribution.

        Returns:
            int: The year of death for the farmer.
        """
        loc = 30-1.5
        c = 0.01

        return gompertz.rvs(c, loc=loc)
        
    def get_consumption(self):
        """
        Calculate and return the total consumption based on the population's age and gender.
        The method iterates through the population list and calculates the consumption
        for each individual based on the following criteria:
        - Individuals older than 14 years:
            - Males consume 321 kg/yr.
            - Females consume 259 kg/yr.
        - Individuals aged 14 years or younger consume 237 kg/yr.
        The total consumption is stored in the `self.consumption` attribute and returned.
        Returns:
            int: The total consumption for the population.
        """
        consumption = 0
        if self.site_type == "Farm":
            for i in range(len(self.population)):
                if self.population[i].age > 14:
                    if self.population[i].gender == 0:
                        consumption += self.random.randint(270, 321)
                    else:
                        consumption += self.random.randint(215, 259)
                else:
                    consumption += self.random.randint(106, 237)
            self.consumption = consumption
        
        return consumption
    
    def population_step(self):
        """
        Simulates a single step in the lifecycle of the farm agent's population. This includes aging, reproduction, death, 
        emancipation, and ensuring the presence of key roles (patriarchs and women) in the population. The method performs 
        the following operations:
        1. **Aging**: Increments the age of each individual in the population.
        2. **Reproduction**: Female individuals (gender = 1) of reproductive age (14 or older) and with fewer than 5 children 
           have a chance to give birth. Births are subject to a probability of occurrence and infant mortality.
        3. **Death**: Removes individuals from the population if their age exceeds their death age.
        4. **Emancipation**: At age 14, individuals are evaluated for emancipation:
           - Female individuals (gender = 1) are removed from the population.
           - Male individuals (gender = 0) may become patriarchs if there are fewer than 2 patriarchs. Otherwise, they are removed.
        5. **Patriarch Check**: Ensures there is at least one patriarch (gender = 0, patriarch = 1) in the population. If none 
           exist, the oldest woman (gender = 1) becomes a patriarch.
        6. **Women Check**: Ensures there is at least one woman (gender = 1, age >= 14) in the population. If none exist, a 
           new woman is added based on the age of the oldest patriarch.
        This method ensures the population maintains a balance of roles and adheres to the rules of aging, reproduction, and 
        mortality.
        """
        if self.site_type == "Farm":
            # Aging
            for i in range(len(self.population)):
                self.population[i].age += 1
                # Reproduction
                if self.population[i].gender == 1 and self.population[i].sons < 5 and self.population[i].age >= 14:
                    if self.random.random() < 0.3: # Probability of giving birth
                        if self.random.random() > 0.4: # Infant mortality
                            self.population.append(self.Farmer(gender=self.random.randint(0, 1), age=0, death_age=self.get_death_date()))
                # Death
                if self.population[i].age >= self.population[i].death_age:
                    del self.population[i]
                    break
                
                # Emancipation
                if self.population[i].gender == 1 and self.population[i].age == 14:
                    del self.population[i]
                    break
                n_patriarch = 0
                for j in range(len(self.population)):
                    if self.population[j].patriarch == 1:
                        n_patriarch += 1
                if self.population[i].gender == 0 and self.population[i].age == 14:
                    if n_patriarch < 2:
                        self.population[i].patriarch = 1
                        self.population.append(self.Farmer(gender=1, age=14, death_age=self.get_death_date()))
                    else:
                        # del self.population[i]
                        break
            # Check if there are any patriarchs in the population
            n_patriarch = 0
            for i in range(len(self.population)):
                if self.population[i].patriarch == 1:
                    n_patriarch += 1
            if n_patriarch == 0:
                oldest_woman = 14
                for j in range(len(self.population)):
                    if self.population[j].gender == 1:
                        oldest_woman = max(oldest_woman, self.population[j].age)
                self.population.append(self.Farmer(gender=0, age=oldest_woman, death_age=self.get_death_date(), patriarch=1))

            if len(self.population) == 1:
                if self.population[0].gender == 0:
                    self.population.append(self.Farmer(gender=1, age=14, death_age=self.get_death_date()))
                else:
                    self.population.append(self.Farmer(gender=0, age=14, death_age=self.get_death_date(), patriarch=1))

    def starve_and_update_area(self):
        """
        Adjusts the population and updates the cultivated area based on wheat stock and consumption.
        This method performs two main tasks:
        1. Removes the youngest farmer from the population if the wheat stock is insufficient
        2. Updates the cultivated area based on the available workforce and wheat stock.
        The cultivated area is calculated considering:
        - The workforce contribution of each farmer (based on age and gender).
        - The available wheat stock relative to the consumption needs.
        - The maximum allowable extension of the cultivated area.

        Returns:
            None
        """
        if self.site_type == "Farm":
            consumption = self.get_consumption()
            area_man = 3.75
            area_woman = 1.75
            man_power = 0
            if consumption > self.wheat:
                index = self.population.index(min(self.population, key=lambda farmer: farmer.age))
                del self.population[index]
            
            # Update area based on the population
            for i in range(len(self.population)):
                if self.population[i].age >= 14:
                    if self.population[i].gender == 0 and self.population[i].age >= 18:
                        man_power += area_man
                    else:
                        man_power += area_woman
            if consumption > self.wheat:
                self.area = min(man_power, self.area, self.maximum_extension)
            else:
                self.area = min(man_power, self.area + (self.wheat - consumption)/self.wheat*self.area, self.maximum_extension)


    def step(self):
        self.production_step()
        self.population_step()
        self.starve_and_update_area()
        self.population_len = len(self.population)