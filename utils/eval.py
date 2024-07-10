import lm_eval
import weave
import wandb
from lm_eval.loggers import WandbLogger
import os
import pathlib


def eval(
    platform: str = "vllm",
    model_name: str = "Llama-2-7b",
    gpu_num: int = 2,
    infer_dtye: str = "auto",
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    batch_size="auto",
    tasks: str = "wikitext",
    log_samples: bool = True,
    use_wandb: bool = True,
    del_logs: bool = False,
):
    model_args = (
        f"pretrained=/data/llmQuantModels/{model_name},"
        + f"tensor_parallel_size={gpu_num},"
        + f"dtype={infer_dtye},"
        + f"gpu_memory_utilization={gpu_memory_utilization},"
        + f"batch_size={batch_size},"
        + f"enforce_eager={enforce_eager},"
    )

    results = lm_eval.simple_evaluate(
        model=platform,
        model_args=model_args,
        tasks=tasks,
        log_samples=log_samples,
    )

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
        wandb_logger = WandbLogger()
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        if log_samples:
            wandb_logger.log_eval_samples(results["samples"])
        if del_logs:
            # search for the log files(wandb/ dir) and delete it
            for path in pathlib.Path("..").rglob("wandb"):
                if os.path.isdir(path):
                    os.system(f"rm -rf {path}")
