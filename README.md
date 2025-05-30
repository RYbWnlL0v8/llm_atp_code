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

### Preparation

##### Create and  activate conda environment

```bash
conda create -n dobevi python=3.11 -y
conda activate dobevi
```

##### Install Python dependencies

```bash
pip install -r requirements.txt
```

##### Install Graphviz dependencies (Required for visualization)

```bash
conda install -c conda-forge graphviz
```

### Evaluation

To run the evaluation, please follow these steps:

##### Prepare a Lean 4 Repository

 You need a Lean 4 project repository that can be built successfully with `lake build`. 

##### Configure Environment Variables

 Copy the environment variable template file . Edit it according to your setup.

```bash
cd src/
cp .env.template .env
```

##### Running Benchmarks

```bash
python -m eval
```