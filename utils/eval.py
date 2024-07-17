import io
import os
import pathlib
import re
from contextlib import redirect_stderr, redirect_stdout

import lm_eval  # type: ignore
from lm_eval.loggers import WandbLogger  # type: ignore

import wandb

# import weave


def eval(
    platform: str = "vllm",
    model_name: str = "Llama-2-7b",
    tensor_parallel: bool = True,
    infer_dtye: str = "auto",
    gpu_memory_utilization: float = 0.7,
    enforce_eager: bool = False,
    batch_size="auto",
    tasks: str = "wikitext",
    log_samples: bool = False,
    use_wandb: bool = True,
    del_logs: bool = False,
    device: str = None,
):
    """Evaluate the model on the given tasks and log the results to wandb.

    Args:
        platform (str, optional):  . Defaults to "vllm".
        model_name (str, optional):  . Defaults to "Llama-2-7b".
        tensor_parallel (bool, optional):  . Defaults to True.
        infer_dtye (str, optional):  . Defaults to "auto".
        gpu_memory_utilization (float, optional):  . Defaults to 0.7.
        enforce_eager (bool, optional): If u want to get the lowest mem overhead(but longer latency), or facing OOM, u can set it to True. Defaults to False.
        batch_size (str, optional):  . Defaults to "auto".
        tasks (str, optional):  . Defaults to "wikitext".
        log_samples (bool, optional):  . Defaults to True.
        use_wandb (bool, optional): (Recommended). Defaults to True.
        del_logs (bool, optional): (Experimental). Defaults to False.
    """

    str_tps = "tensor_parallel_size=2," if tensor_parallel else ""
    gpu_num = 2 if tensor_parallel else 1

    # device = device if device else "cuda:0"
    # str_device = "" if tensor_parallel else f"device={device},"

    model_args = (
        f"pretrained=/data/llmQuantModels/{model_name},"
        + f"dtype={infer_dtye},"
        + f"gpu_memory_utilization={gpu_memory_utilization},"
        + f"batch_size={batch_size},"
        + f"enforce_eager={enforce_eager},"
        + str_tps
        # + str_device
    )

    llm_eval_out = io.StringIO()
    llm_eval_err = io.StringIO()
    with redirect_stdout(llm_eval_out), redirect_stderr(
        llm_eval_err
    ):  # Redirect the stdout and stderr to the io.StringIO object
        results = lm_eval.simple_evaluate(
            model=platform,
            model_args=model_args,
            tasks=tasks,
            log_samples=log_samples,
            device=device,
        )

    # Analyze the log
    log_out = llm_eval_out.getvalue() + llm_eval_err.getvalue()
    if platform == "vllm":
        model_size, avg_output_toks = parse_log_vllm(log_out)
        results["model_size"] = model_size
        results["avg_output_toks"] = avg_output_toks
    if not del_logs:
        with open(f"log/{model_name}_on{gpu_num}.log", "w+") as f:
            f.write(log_out)

    # Log results to wandb
    if use_wandb and wandb.login():
        wandb.init(
            project="llm_quant_tests",
            job_type="eval",
            config={  # hyperparameters you want to track
                "platform": platform,
                "model_name": model_name,
                "gpu_num": gpu_num,
                "dtype": infer_dtye,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
                "batch_size": batch_size,
                "tasks": tasks,
            },
        )
        wandb.log(
            {
                "model size": results["model_size"],
                "avg_output_toks": results["avg_output_toks"],
            }
        )
        wandb_logger = WandbLogger()
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        if log_samples:
            wandb_logger.log_eval_samples(results["samples"])
        if del_logs:
            # search for the log files(wandb/ dir) and delete it
            for path in pathlib.Path("..").rglob("wandb"):
                if os.path.isdir(path):
                    os.system(f"rm -r {path}")

        return results


def parse_log_vllm(log_content):
    # Extract model size
    size_match = re.search(r"Loading model weights took (\d+\.\d+) GB", log_content)
    model_size = float(size_match.group(1)) if size_match else None

    # Extract output tokens/s
    output_toks_matches = re.findall(r"output: (\d+\.\d+) toks/s", log_content)
    output_toks = [float(rate) for rate in output_toks_matches]
    # drop the 0.0 toks/s
    output_toks = [rate for rate in output_toks if rate > 0]
    avg_output_toks = sum(output_toks) / len(output_toks) if output_toks else None

    return model_size, avg_output_toks
