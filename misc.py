from typing import Callable
import pprint

import uproot
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

import mplhep

import constant

plt.style.use(mplhep.style.LHCbTex2)

def format_number(num):
    # Check if number is very close to an integer (within floating point precision)
    if abs(num) >= 1:
        if abs(round(num) - num) < 1e-8:
            return f"{round(num)}"
        return f"{num:.2f}" if not num.is_integer() else f"{num:.0f}"
    else:
        return f'{float(f"{num:.3g}"):g}'


class HistogramPlot:
    def __init__(
        self,
        series: pd.Series,
        label: str,
        error: bool = None,
        bins: float | np.ndarray = None, # bin width or bin edges
        **kwargs,  # plot kwargs
    ):
        self.series = series
        self.label = label
        self.error = error
        self.kwargs = kwargs

        if isinstance(bins, np.ndarray):
            bins = bins
        elif isinstance(bins, float) or isinstance(bins, int):
            bins = self.calculate_bins(self.series, bins)
        else:
            bins = "fd"

        self.h, self.bin_edges = np.histogram(series, bins=bins)

    def calculate_bins(_self, series: pd.Series, bin_width: int | float):
        return np.arange(series.min(), series.max() + bin_width, bin_width)
    
    @property
    def bin_width(self) -> float:
        return self.bin_edges[1] - self.bin_edges[0]

def histogram_ax(
    hist_plots,
    ax,
    density: bool = False,
    legend: bool = True,
    grid: bool = False,
    xlabel: str | None = None,
    unit_tex: None | str = None,
    **ax_kwargs,
):
    ax.set(**ax_kwargs)

    # plots
    for hist_plot in hist_plots:
        mplhep.histplot(
            hist_plot.h,
            hist_plot.bin_edges,
            ax=ax,
            label=hist_plot.label,
            yerr=hist_plot.error,
            density=density,
            **hist_plot.kwargs,
        )

    # set axis labels
    if density:
        ylabel = f"Candidates (Normalised)"
    elif len({hist_plot.bin_width for hist_plot in hist_plots}) == 1 and unit_tex is not None:
        bin_width = hist_plots[0].bin_width
        ylabel = f"Candidates / ({format_number(bin_width)}" + f" ${unit_tex}$)"
    else:
        bin_width = hist_plots[0].bin_width
        ylabel = f"Candidates / {format_number(bin_width)}"

    if unit_tex is not None:
        if xlabel is None:
            xlabel = f"${unit_tex}$"
        else:
            xlabel = " ".join((xlabel.strip(), f"[${unit_tex}$]"))

    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.set(**ax_kwargs)
    if legend:
        ax.legend()
    if grid:
        ax.grid()


def histogram_fig(
    hist_plots: list[HistogramPlot],
    save_to_file: str | bool,
    preset: str | None = None,
    **ax_kwargs,
):
    """
    Wrapper around histogram_ax with save to file functionality
    """
    fig, ax = plt.subplots()

    if preset == "Bplus Mass":
        ax_kwargs["unit_tex"] = r"\mathrm{MeV}/c^2"
        ax_kwargs["xlabel"] = "Mass"
        ax_kwargs["xlim"] = [constant.BPLUS_MASS - 100, constant.BPLUS_MASS + 100]

    histogram_ax(hist_plots, ax, **ax_kwargs)

    # save to file
    if save_to_file:
        fig.savefig(save_to_file)
        print(f"Histogram saved to {save_to_file}")
    else:
        print("Histogram not saved to file")        

    return fig, ax

def find_distances(r1, r2):
    return np.linalg.norm(np.array(r1) - np.array(r2))

def save_root(df: pd.DataFrame, path: str, tree_name: str):
    output_columns = {}
    for series_name in df.columns:
        if df[series_name].dtype != "awkward":
            output_columns[series_name] = df[series_name].to_numpy()

    with uproot.recreate(path) as root_file:
        root_file[tree_name] = output_columns

    print(f'"{path}" saved')


def load_root(path: str, tree_name: str | None = None) -> pd.DataFrame:
    with uproot.open(path) as file:
        if tree_name is None:
            print("No tree name was given. The avaliable keys are:")
            pprint.pprint(file.keys())
        else:
            ttree = file[tree_name]
            df = ttree.arrays(library="pd")
            return df
        
def filter_duplicates_by_performance(
    df: pd.DataFrame, performance_column: str, keep: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate candidates based on the performance column.
    """
    if keep.lower() == "min":
        ascending = True
    elif keep.lower() == "max":
        ascending = False
    else:
        raise ValueError("'keep' parameter must be 'min' or 'max'")

    sorted_df = df.sort_values(performance_column, ascending=ascending)

    # Keep the best performing row from each set of duplicates
    unique_candidates_mask = ~sorted_df.duplicated(
        subset=["runNumber", "eventNumber"], keep="first"
    )

    return sorted_df[unique_candidates_mask], sorted_df[~unique_candidates_mask]

def save_root(df: pd.DataFrame, path: str, tree_name: str):
    output_columns = {}
    for series_name in df.columns:
        if df[series_name].dtype != "awkward":
            output_columns[series_name] = df[series_name].to_numpy()

    with uproot.recreate(path) as root_file:
        root_file[tree_name] = output_columns

    print(f'"{path}" saved')

def ax_units_pi(ax, scale=0.1):
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: "{:.2g}".format(val / np.pi) + r"$\pi$" if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi*scale))
    return ax


def log_pT2(p_T):
    return np.log(p_T**2)

def p_T(log_pT2):
    return np.sqrt(np.exp(log_pT2))

def add_log_pt_axis(ax, unit="MeV"):
    ax2 = ax.secondary_xaxis("top", functions=(p_T,log_pT2))

    ax2.set_xscale("log")
    ax2.set_xlabel(rf"$p_T$ [$\mathrm{{{unit}}}/c$]")

    ax.tick_params(top=False, labeltop=False)
    ax.xaxis.set_ticks_position('bottom')