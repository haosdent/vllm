# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import lm_eval
import openai

BASE_URL = "http://localhost:8192/v1"
# [D43301] Sequential execution to make L4 and A30 logs comparable
# step-by-step. Bug may not reproduce at concurrent=1 -- if so we know
# concurrency is involved and re-run with larger concurrency.
NUM_CONCURRENT = 1
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
# [D43301] Limit to N prompts so logs are scannable. 50 still has enough
# samples to see L4 drop pattern if it's per-prompt-deterministic.
GSM8K_LIMIT = 50

# Model-specific expected values
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.41,
    "deepseek-ai/deepseek-vl2-small": 0.59,
    "deepseek-ai/deepseek-vl2-tiny": 0.19,
    "deepseek-ai/DeepSeek-V2-Lite-Chat": 0.65,
    "google/gemma-3-4b-it": 0.74,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": 0.84,
    "ibm-granite/granite-4.0-h-tiny": 0.80,
    "Qwen/Qwen3.5-0.8B": 0.33,
}

SIMPLE_PROMPT = (
    "The best part about working on vLLM is that I got to meet so many people across "
    "various different organizations like UCB, Google, and Meta which means",
)

# Get model name from environment variable
MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")


def run_simple_prompt():
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    completion = client.completions.create(model=MODEL_NAME, prompt=SIMPLE_PROMPT)

    print("-" * 50)
    print(f"Completion results for {MODEL_NAME}:")
    print(completion)
    print("-" * 50)


def test_accuracy():
    """Run the end to end accuracy test."""
    run_simple_prompt()

    model_args = (
        f"model={MODEL_NAME},"
        f"base_url={BASE_URL}/completions,"
        f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
    )

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
        limit=GSM8K_LIMIT,
        # [D43301] force greedy so chosen=top1; otherwise sampler picks
        # stochastically and we can't compare L4/A30 chosen-token streams.
        gen_kwargs="temperature=0,do_sample=False,top_k=1",
    )

    measured_value = results["results"][TASK][FILTER]
    expected_value = EXPECTED_VALUES.get(MODEL_NAME)

    print(
        f"[D43301-RESULT] model={MODEL_NAME} expected={expected_value} "
        f"measured={measured_value}"
    )

    if expected_value is None:
        print(
            f"Warning: No expected value found for {MODEL_NAME}. "
            "Skipping accuracy check."
        )
        return

    assert (
        measured_value - RTOL < expected_value
        and measured_value + RTOL > expected_value
    ), f"Expected: {expected_value} | Measured: {measured_value}"
