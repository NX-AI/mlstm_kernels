#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import gc
import logging
import pprint
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import yaml

from .benchmarks.interface import BenchmarkCreator, ModelBenchmarkCreator
from .param_handling import BenchmarkConfig
from .plot_results import plot_benchmark_result_table

LOGGER = logging.getLogger(__name__)


def run_benchmarks(
    benchmark_config: BenchmarkConfig,
    benchmark_creator: BenchmarkCreator,
    param_prefix: str = "P--",
    additional_param_name_short: bool = True,
    runtime_prefix: str = "R--",
    memory_prefix: str = "M--",
    output_folder: Path = None,
    run_garbage_collection: bool = True,
) -> pd.DataFrame:
    """Runs the different kernel configurations and summarizes the results in a DataFrame.

    Args:
        benchmark_config: The benchmark configuration.
        param_prefix: The prefix to add to the parameter names in the DataFrame.
    """
    param_dicts = benchmark_config.get_param_dicts()
    kernel_specs = benchmark_config.kernel_specs
    results = []
    for i, param_dict in enumerate(param_dicts):
        LOGGER.info(f"Parameter combination ({i+1}/{len(param_dicts)}): {param_dict}")
        # add a prefix to easily identify the parameters in the DataFrame
        result_dict = {f"{param_prefix}{k}": v for k, v in param_dict.items()}
        for k, kernel_spec in enumerate(kernel_specs):
            benchmark = benchmark_creator(kernel_spec, param_dict)
            benchmark.set_params(kernel_spec.additional_params)
            benchmark.setup_benchmark()
            runtime_results = benchmark.run_benchmark()
            result_dict[
                f"{runtime_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.runtime
            result_dict[
                f"{memory_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.peak_memory_allocated
            LOGGER.info(
                (
                    f"Kernel ({k+1}/{len(kernel_specs)}): {kernel_spec.to_string()} finished.",
                    f" Runtime: {runtime_results.runtime} ms. Peak memory: {float(runtime_results.peak_memory_allocated / 10**9)} GB.",
                )
            )
            del benchmark
            if run_garbage_collection:
                gc.collect()
                torch.cuda.empty_cache()
        results.append(result_dict)
        if output_folder is not None:
            result_df = pd.DataFrame(results)
            result_df.to_csv(output_folder / "results.csv")

    return pd.DataFrame(results)


def run_model_benchmarks(
    benchmark_config: BenchmarkConfig,
    benchmark_creator: ModelBenchmarkCreator,
    param_prefix: str = "P--",
    additional_param_name_short: bool = True,
    runtime_prefix: str = "R--",
    memory_prefix: str = "M--",
    setup_model_on_every_param_combination: bool = False,
    profiler=None,
    output_folder: Path = None,
    run_garbage_collection: bool = True,
) -> pd.DataFrame:
    """Runs the different model configurations and summarizes the results in a DataFrame.
    This differs from the kernel benchmark in that way that the two loops are switched.
    The reason for this is that for model benchmarks the model loading can take a long time.
    So we want to do that once for every parameter combination and then run the different kernels.
    """
    param_dicts = benchmark_config.get_param_dicts()
    kernel_specs = benchmark_config.kernel_specs
    results = [
        {f"{param_prefix}{k}": v for k, v in param_dict.items()}
        for param_dict in param_dicts
    ]
    for k, kernel_spec in enumerate(kernel_specs):
        LOGGER.info(f"Model ({k+1}/{len(kernel_specs)}): {kernel_spec.to_string()}")
        if not setup_model_on_every_param_combination:
            benchmark = benchmark_creator(kernel_spec, kernel_spec.additional_params)
            LOGGER.info(f"Created benchmark: \n{pprint.pformat(benchmark)}")
            benchmark.setup_model()

        for i, param_dict in enumerate(param_dicts):
            if setup_model_on_every_param_combination:
                benchmark = benchmark_creator(
                    kernel_spec, kernel_spec.additional_params
                )
                benchmark.setup_model()

            LOGGER.info(
                f"Parameter combination ({i+1}/{len(param_dicts)}): {param_dict}"
            )
            # add a prefix to easily identify the parameters in the DataFrame
            # result_dict = {f"{param_prefix}{k}": v for k, v in param_dict.items()}

            benchmark.set_params(param_dict)
            benchmark.setup_benchmark()
            runtime_results = benchmark.run_benchmark(profiler=profiler)
            results[i][
                f"{runtime_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.runtime
            results[i][
                f"{memory_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.peak_memory_allocated
            LOGGER.info(
                (
                    f"Parameter combination ({i+1}/{len(param_dicts)}) finished.",
                    f" Runtime: {runtime_results.runtime} ms.",
                    f" Peak memory: {float(runtime_results.peak_memory_allocated / 10**9)} GB.",
                )
            )
            if setup_model_on_every_param_combination:
                del benchmark
            if run_garbage_collection:
                gc.collect()
                torch.cuda.empty_cache()
            if output_folder is not None:
                result_df = pd.DataFrame(results)
                result_df.to_csv(output_folder / "results.csv")

        del benchmark
        if run_garbage_collection:
            gc.collect()
            torch.cuda.empty_cache()

    LOGGER.info("Finished all benchmarks.")
    return pd.DataFrame(results)


def run_and_record_benchmarks(
    benchmark_config: BenchmarkConfig,
    benchmark_creator: BenchmarkCreator,
    output_folder: Path,
    benchmark_type: Literal["model", "kernel"] = "kernel",
    **kwargs,
):
    import logging

    import tabulate

    LOGGER = logging.getLogger(__name__)

    LOGGER.info(f"Running benchmark: {benchmark_config.benchmark_name}")

    benchmark_folder = output_folder / benchmark_config.benchmark_name
    benchmark_folder.mkdir(parents=True, exist_ok=False)

    with (benchmark_folder / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(asdict(benchmark_config), config_file, sort_keys=False)

    if benchmark_type == "kernel":
        run_benchmarks_fn = run_benchmarks
    elif benchmark_type == "model":
        run_benchmarks_fn = run_model_benchmarks
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    result_df = run_benchmarks_fn(
        benchmark_config=benchmark_config,
        benchmark_creator=benchmark_creator,
        additional_param_name_short=True,
        output_folder=benchmark_folder,
        **kwargs,
    )

    LOGGER.info(
        f"Results:\n{tabulate.tabulate(result_df, headers='keys', tablefmt='pretty')}"
    )
    LOGGER.info(f"Saving results to {benchmark_folder}")
    result_df.to_csv(benchmark_folder / "results.csv")

    def plot_result_table(
        additional_exclude_col_regex: str, plot_name_suffix: str, y_label: str
    ):
        fig = plot_benchmark_result_table(
            result_df,
            benchmark_config.x_axis_param,
            title=f"Runtime--{benchmark_config.get_plot_title()}",
            additional_exclude_col_regex=additional_exclude_col_regex,
            y_label=y_label,
        )

        def savefig(file_ending):
            fig.savefig(
                benchmark_folder
                / f"plot_{benchmark_config.benchmark_name}_{plot_name_suffix}.{file_ending}",
                dpi=300,
                bbox_inches="tight",
            )

        for file_ending in ["png", "pdf", "svg"]:
            savefig(file_ending)

    # runtime plot
    plot_result_table("M--.*", "runtime", "Time [ms]")
    # memory plot
    plot_result_table("R--.*", "memory", "Memory [bytes]")
