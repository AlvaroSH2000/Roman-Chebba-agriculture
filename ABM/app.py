from model import ChebbaFarms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import solara
from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter
import geopandas as gpd
import pandas as pd


def _agent_df(model):
    return model.datacollector.get_agent_vars_dataframe().reset_index()


def _extract_grid_arrays(voronoi_data):
    nx = int(voronoi_data.get("nx", 0))
    ny = int(voronoi_data.get("ny", 0))

    if "mask" in voronoi_data and (nx == 0 or ny == 0):
        flat_mask = np.asarray(voronoi_data["mask"])
        if nx == 0 and "x" in voronoi_data:
            nx = int(np.asarray(voronoi_data["x"]).size)
        if ny == 0 and "y" in voronoi_data:
            ny = int(np.asarray(voronoi_data["y"]).size)
        if nx == 0 or ny == 0:
            side = int(np.sqrt(flat_mask.size))
            nx, ny = side, side

    x_grid = voronoi_data.get("x_grid")
    y_grid = voronoi_data.get("y_grid")
    if x_grid is not None and y_grid is not None:
        return np.asarray(x_grid), np.asarray(y_grid), ny, nx

    x_center = voronoi_data.get("x_center")
    y_center = voronoi_data.get("y_center")
    if x_center is not None and y_center is not None:
        return np.asarray(x_center), np.asarray(y_center), ny, nx

    if "x" in voronoi_data and "y" in voronoi_data:
        x_vals = np.asarray(voronoi_data["x"])
        y_vals = np.asarray(voronoi_data["y"])
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        return x_grid, y_grid, y_vals.size, x_vals.size

    return None, None, ny, nx


def _plot_villa_metric(fig, model, metric_name):
    ax = fig.subplots()
    agent_df = _agent_df(model)
    villas = agent_df[agent_df["AgentType"] == "VillaAgent"].copy()
    if villas.empty:
        ax.set_title(f"Evolucio temporal VillaAgents - {metric_name}")
        ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=ax.transAxes)
        return

    villas = villas[["Step", "AgentID", metric_name]].dropna()
    if villas.empty:
        ax.set_title(f"Evolucio temporal VillaAgents - {metric_name}")
        ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=ax.transAxes)
        return

    metric_by_villa = villas.pivot_table(index="Step", columns="AgentID", values=metric_name, aggfunc="last")
    ax.plot(metric_by_villa.index, metric_by_villa.values)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Evolucio temporal VillaAgents - {metric_name}")
    ax.grid(True, alpha=0.2)


def _plot_climate_evolution(fig, model):
    ax = fig.subplots()
    model_df = model.datacollector.get_model_vars_dataframe().reset_index(names="Step")
    if model_df.empty:
        ax.set_title("Evolucio temporal del clima")
        ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=ax.transAxes)
        return

    temp_map = {"cool": 1, "normal": 2, "warm": 3, "very_warm": 4}
    rain_map = {"very_dry": 1, "dry": 2, "normal": 3, "humid": 4}

    climate_df = model_df[["Step", "Temperature", "Rain"]].copy()
    climate_df["TemperatureNum"] = climate_df["Temperature"].map(temp_map)
    climate_df["RainNum"] = climate_df["Rain"].map(rain_map)

    valid_df = climate_df.dropna(subset=["TemperatureNum", "RainNum"])
    if valid_df.empty:
        ax.set_title("Evolucio temporal del clima")
        ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(valid_df["Step"], valid_df["TemperatureNum"], marker="o", label="Temperature")
    ax.plot(valid_df["Step"], valid_df["RainNum"], marker="s", label="Rain")

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Categoria (1-4)")
    ax.set_title("Evolucio temporal del clima")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")



def _plot_field_2d(fig, model):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    
    gdf = gpd.read_file("input_data/chebba_clipped.shp")
    voronoi_gdf = gpd.read_file("input_data/voronoi_regions.shp")
    sites = pd.read_csv("input_data/sites_voronoi.csv")

    ax = fig.subplots()
    agent_df = _agent_df(model)
    fields = agent_df[(agent_df["AgentType"] == "FieldAgent") & (agent_df["Step"] == agent_df["Step"].max())].copy()
    if fields.empty:
        ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=ax.transAxes)
        return

    x_grid, y_grid, ny, nx = _extract_grid_arrays(model.voronoi)
    if ny <= 0 or nx <= 0:
        ax.text(0.5, 0.5, "Grid invalid", ha="center", va="center", transform=ax.transAxes)
        return

    # Create custom colormaps
    cmap_qv = LinearSegmentedColormap.from_list("plum_darkmagenta", ["plum", "darkmagenta"])
    cmap_qo = LinearSegmentedColormap.from_list("yellowgreen_darkolivegreen", ["yellowgreen", "darkolivegreen"])
    cmap_qw = LinearSegmentedColormap.from_list("lightgoldenrodyellow_gold", ["lightgoldenrodyellow", "gold"])
    
    # Define metrics with their colormaps and transparency
    metrics = [
        ("Q_v", cmap_qv, 1),
        ("Q_o", cmap_qo, 1),
        ("Q_w", cmap_qw, 1),
    ]

    for metric, cmap, alpha in metrics:
        grid = np.full((ny, nx), np.nan, dtype=float)
        flat = grid.ravel()

        for row in fields.itertuples(index=False):
            idx = int(row.ID)
            if 0 <= idx < flat.size:
                val = getattr(row, metric, np.nan)
                if val == 0:
                    val = np.nan
                flat[idx] = val

        finite_vals = grid[np.isfinite(grid)]
        positive_vals = finite_vals[finite_vals > 0]

        if x_grid is not None and y_grid is not None and positive_vals.size > 0:
            norm = Normalize(vmin=float(positive_vals.min()), vmax=float(positive_vals.max()))
            ax.pcolormesh(
                x_grid,
                y_grid,
                grid,
                cmap=cmap,
                norm=norm,
                shading="nearest",
                alpha=alpha,
            )

    voronoi_gdf.boundary.plot(ax=ax, color="grey", linewidth=1)
    gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    ax.scatter(sites["x"], sites["y"], color='k', s=5, label='Sites')
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")


@solara.component
def VillaMetricPlot(model, metric_name):
    update_counter.get()
    fig = Figure(figsize=(10, 5), constrained_layout=True)
    _plot_villa_metric(fig, model, metric_name)
    solara.FigureMatplotlib(fig)


@solara.component
def Field2DPlot(model):
    update_counter.get()
    fig = Figure(figsize=(9, 8), constrained_layout=True)
    _plot_field_2d(fig, model)
    solara.FigureMatplotlib(fig)


@solara.component
def ClimateEvolutionPlot(model):
    update_counter.get()
    fig = Figure(figsize=(10, 5), constrained_layout=True)
    _plot_climate_evolution(fig, model)
    solara.FigureMatplotlib(fig)


def make_villa_metric_component(metric_name):
    @solara.component
    def _component(model):
        VillaMetricPlot(model, metric_name)

    return _component


def make_field_2d_component():
    @solara.component
    def _component(model):
        Field2DPlot(model)

    return _component



model_params = {
    "seed": {
        "type": "InputText",
        "value": 1,
        "label": "Random Seed",
    }
}

model = ChebbaFarms(seed=int(model_params["seed"]["value"]))

page = SolaraViz(
    model,
    components=[
        ClimateEvolutionPlot,
        make_villa_metric_component("Q_v"),
        make_villa_metric_component("Q_o"),
        make_villa_metric_component("Q_w"),
        # make_field_2d_component(),
    ],
    model_params=model_params,
    name="Chebba Farms Model",
)
