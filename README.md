# ReflectScore

ReflectScore is a reproducible benchmark for measuring how reflection agents affect LLM hallucination across factual questions, code-grounded retrieval tasks, and unanswerable prompts.

The checked-in report, plots, and result artifacts in this repository correspond to the completed v1 benchmark run.

## What It Measures

ReflectScore compares six systems:

1. Baseline
2. Forced Reflection
3. Confidence-Triggered Reflection
4. Self-Reflection
5. Cross-Agent Reflection
6. Reflection + Verifier

The benchmark is designed to study five under-measured research gaps:

- Whether forced reflection consistently reduces hallucination
- When reflection backfires and makes correct answers worse
- Whether confidence-triggered reflection saves API calls without losing quality
- Whether peer review outperforms self-review
- Where iterative refinement plateaus or degrades over multiple rounds

## Project Layout

```text
reflectscore/
|- .env
|- config.py
|- run_benchmark.py
|- retrieval.py
|- data/
|- systems/
|- evaluation/
|- report/
|- tests/
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add a real Google AI Studio key to `.env`:

```env
GEMINI_API_KEY=your_key_here
MODEL=gemini-2.0-flash
TEMPERATURE=0.3
MAX_ITERATIONS=3
SEED=42
REQUESTS_PER_MINUTE=15
RATE_LIMIT_COOLDOWN_SECONDS=10
```

## How To Run

Run the full live benchmark:

```bash
python run_benchmark.py
```

Run a smoke benchmark with the first 5 questions from each dataset:

```bash
python run_benchmark.py --smoke
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Generate the markdown report from existing live outputs:

```bash
python report/generate_report.py
```

## Retrieval Configuration

- Chunk size: 200 tokens
- Overlap: 50 tokens
- Top-k retrieval: 3 chunks
- Chunking uses the MiniLM tokenizer when available, with a whitespace fallback only for tokenizer unavailability
- Shared retrieved context is used for all systems on `code` and `unanswerable` questions for fairness

## Metrics

- Hallucination Rate: fraction of final answers that fail automated correctness checks
- Grounding Score: fraction of code answers that cite a real function or file from the context
- Refusal Accuracy: fraction of unanswerable questions that are correctly refused
- Correction Rate: fraction of reflective cases where a wrong initial answer becomes correct
- Backfire Rate: fraction of reflective cases where a correct initial answer becomes wrong
- Confidence Calibration: alignment between predicted confidence and actual correctness
- Mean Latency: average response time per system in seconds

## Outputs

- `results/raw_results.csv`: final answer records per question and system
- `results/iteration_results.csv`: intermediate answer snapshots by iteration
- `results/summary.json`: metrics, confidence intervals, and latency summary
- `visualizations/*.png`: benchmark plots
- `report/benchmark_report.md`: markdown research summary

## Run History

- v1: llama-3.1-8b-instant, Groq, 50 questions, completed
- v2: llama-3.3-70b-versatile, Groq, blocked by rate limits
- v3: gemini-2.0-flash, Google AI Studio, 80 questions, in progress

## Sample Results Table

| System | Hallucination Rate | Grounding | Refusal | Backfire | Mean Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.32 | 0.45 | 0.20 | - | 1.12 |
| Forced Reflection | 0.21 | 0.61 | 0.35 | 0.08 | 3.84 |
| Confidence Reflection | 0.18 | 0.65 | 0.42 | 0.05 | 2.41 |
| Self Reflection | 0.19 | 0.63 | 0.38 | 0.06 | 2.76 |
| Cross-Agent Reflection | 0.14 | 0.71 | 0.55 | 0.04 | 4.22 |
| Verifier Reflection | 0.07 | 0.88 | 0.90 | 0.02 | 5.09 |

## Notes

- Runs are seeded with `SEED=42` for reproducibility.
- Confidence intervals use seeded bootstrap resampling with 1000 draws.
- All LLM requests flow through `systems/llm.py`.
- The benchmark does not include any runtime mock LLM mode. A real Google AI Studio API key is required for live execution.
- The iteration curve is computed from `forced_reflection` snapshots so the plot reflects one consistent iterative policy instead of mixing heterogeneous systems.
