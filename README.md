# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation

This repository contains the implementation for the paper *"LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation"*

## Directory Structure

```text
DoBeVi/
├── requirements.txt         # Python dependencies
└── src/                     # Source code
    ├── __init__.py          # Package initializer
    ├── dojo/                # Minimal LeanDojo implementation for interacting with Lean4 kernel
    ├── search/              # Proof search algorithms and logic
    ├── eval.py              # Entry point for evaluation
    ├── config.py            # Configuration settings 
    ├──  utils.py            # Utility functions
    ├── .env                 # Example environment variables file
    └── .env.template        # Template for environment variables
Visual/                      # Contains some proof search tree visualization images.
README.md                    # Project documentation
```

## Preparation

### Create and Activate Conda Environment

```bash
conda create -n dobevi python=3.11 -y
conda activate dobevi
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Graphviz Dependencies (Required for Visualization)

```bash
conda install -c conda-forge graphviz
```

### Download the Policy Model

Please download the policy model from the [HuggingFace page](https://huggingface.co/RYbWnlL0v8/llm_atp_model).

## Evaluation

To run the evaluation, please follow these steps:

### Prepare a Lean 4 Repository

You need a Lean 4 project repository that can be built successfully with `lake build`. 

### Configure Environment Variables

Copy the environment variable template file. In the `.env` file, you need to configure settings according to your setup and evaluation requirements. This includes specifying the benchmark path, model path, budget for the tree search process, and the path for saving evaluation results. Refer to the template for more details

```bash
cd src/
cp .env.template .env
```

### Running Benchmarks

```bash
python -m eval
```