# Sets of Languages

This repository contains the code for the paper "Benchmarking Neural Networks
on Formal Language Classes," which introduces a benchmark on formal language
classes called SoL (Sets of Languages). This repository includes all of the
code necessary to reproduce the experiments and figures in the paper, as well
as a Docker image definition that can be used to replicate the software
environment it was developed in.

## Directory Structure

* `scripts/`: Helper scripts for setting up the software environment, building
  container images, running containers, installing Python packages, etc.
  Instructions for using these scripts are below.
* `src/recognizers/`: Python and Bash source code for sampling languages,
  generating datasets, training and evaluating neural networks, analyzing
  results, etc.
  * `theory/`: Various data structures and algorithms related to the theory of
    computing.
  * `calibration/`: Code for calibrating the hyperparameters of the language
    distributions using a baseline classifier.
  * `language_sampling/`: Code for randomly sampling languages.
  * `hand_picked_languages/`: Implementations of certain hand-picked languages
    (as opposed to randomly sampled languages).
  * `string_sampling/`: Code for sampling positive strings from languages and
    checking string membership, including preprocessing steps.
  * `dataset_generation/`: Code for generating datasets from languages.
  * `neural_networks/`: Code for training and evaluating neural networks.
  * `analysis/`: Code for analyzing results and generating figures.
  * `tools/`: Miscellaneous functions and data structures.
* `tests/`: Unit tests for the code under `src/`, mostly in PyTest.
* `experiments/`: Contains high-level scripts for reproducing all of the
  experiments and figures in the paper.

## Installation and Setup

The file `experiments/submit_job.bash` is a stub that needs to be edited to
submit jobs to one's own computing cluster. The other scripts under
`experiments/` depend on it to submit jobs to run experiments.

Next, run the following script to submit a job that will install all Python
dependencies. The reason it should be submitted as a job is that numpy needs to
be compiled from source, which should not be done on a frontend node.

    $ bash experiments/setup/submit_job.bash

Assuming that runs successfully, you are now ready to run experiments. You can
quickly test this by submitting a job that runs all unit tests.

    $ bash experiments/tests/submit_unit_tests_job.bash

The code should be run inside a Singularity container and inside a virtualenv
managed by Poetry that manages the necessary Python packages. The scripts under
`experiments/` should take care of this automatically. You can manually open a
shell inside the Singularity container by running the following.

    $ bash experiments/singularity_shell.bash

Then, you can open yet another sub-shell inside the Poetry virtualenv by
running the following.

    $ bash scripts/poetry_shell.bash

## Local Installation and Setup

In order to foster reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile-dev`](Dockerfile-dev). To run this code, you can build the Docker
image yourself and run it using Docker. Or, if you don't feel like installing
Docker, you can simply use `Dockerfile-dev` as a reference for setting up the
software environment on your own system. You can also build an equivalent
[Singularity](https://sylabs.io/docs/#singularity) image which can be used on an
HPC cluster, where it is likely that Docker is not available but Singularity is.
There is a script that automatically sets up the Docker container and opens a
shell in it (instructions below).

### Using Docker

In order to use the Docker image, you must first
[install Docker](https://www.docker.com/get-started). If you intend to run any
experiments on a GPU, you must also ensure that your NVIDIA driver is set up
properly and install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
Our experiments are quite fast even on CPU.

In order to automatically build the Docker image, start the container, and open
up a bash shell inside of it, run

    $ bash scripts/docker_shell.bash --build

After you have built the image once, there is no need to do so again, so
afterwards you can simply run

    $ bash scripts/docker_shell.bash

By default, this script starts the container in GPU mode, which will fail if
you are not running on a machine with a GPU. If you only want to run things in
CPU mode, you can run

    $ bash scripts/docker_shell.bash --cpu

### Using Singularity

Singularity is an alternative container runtime that is more suitable for
shared computing environments. Note: Singularity also goes by the name
Apptainer; they are forks of the same project.

In order to build the Singularity image, you must first obtain the Docker image
and then convert it to a `.sif` (Singularity image) file on a machine where you
have root access (for example, your personal computer or workstation). This
requires installing both Docker and
[Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html)
on that machine. Assuming you have already built the Docker image according to
the instructions above, you can use the following to create the `.sif` file:

    $ bash scripts/build_singularity_image.bash

This will create the file `sets-of-languages.sif`. It is normal for this to take
several minutes. Afterwards, you can upload the `.sif` file to your HPC cluster
and use it there.

You can open a shell in the Singularity container using

    $ bash scripts/singularity_shell.bash

This will work on machines that do and do not have an NVIDIA GPU, although it
will output a warning if there is no GPU.

### Additional Setup

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or no container), you must run this script once (*inside
the container shell*):

    $ bash scripts/setup.bash

Specifically, this script installs the Python packages required by our code,
which will be stored in the local directory rather than system-wide.

## Adding New Python Packages

We use the package manager [Poetry](https://python-poetry.org/) to manage Python
packages. It's like pip, but you don't need to manually update a
`requirements.txt` file. Poetry tracks all of the Python packages required by
the code in the files `pyproject.toml` and `poetry.lock`. In order to add a new
Python package, run `poetry add <package-name>`, and commit the updated
`pyproject.toml` and `poetry.lock` files to git.

The Poetry configuration used by GitHub actions should also be updated at the
same time. This is automated by the script `scripts/update_github_poetry.bash`.

## Running Code

All files under `src/` should be run using `poetry` so they have access to the
Python packages provided by the Poetry package manager. This means you should
either prefix all of your commands with `bash scripts/poetry_run.bash` or run
`bash scripts/poetry_shell.bash` beforehand to enter a shell with Poetry's
virtualenv enabled all the time. You should run both Python and Bash scripts
with Poetry, because the Bash scripts might call out to Python scripts. All code
under `src/` should be run with `src/` as the current working directory.

All scripts under `scripts/` and `experiments/` should be run with the
top-level directory as the current working directory.

## Running Experiments

The [`experiments/`](experiments) directory contains scripts for reproducing all
of the experiments and figures in the paper. They should be run outside of the
container. Some of these scripts are intended to be used to submit jobs to a
computing cluster. You can configure the base directory where all results are
written by setting `BASE_DIR` in a file named `ignore/config_override.bash`.

All random sampling is done with fixed random seeds, so the results are
deterministic. The only exception is the dropout masks used when training neural
networks, so there may be minor variation in training runs.

### Calibration

First, we need to calibrate the hyperparameters of the distribution over
languages for each class. These scripts are under `experiments/calibration/`.

* `generate_language_distributions.bash`: Randomly generate many different
  hyperparameter settings for each class.
* `submit_jobs.bash`: Submit jobs to sample many languages for each
  hyperparameter setting, generate datasets from them, and calculate baseline
  accuracy.
* `get_calibrated_distributions.bash`: Select the hyperparameter settings that
  cause the language classes to have the closest baseline accuracy.

### Dataset Generation

Next, using the selected hyperparameters, we need to sample languages from each
class and generate datasets from them. These scripts are under
`experiments/dataset_generation/`.

Dataset generation consists of the following steps:

1. Randomly sample a language as a DFA or CFG and write it to a .pt file.
2. Preprocess the DFA or CFG so it can be used for efficient length-constrained
   sampling and membership testing.
3. Randomly sample training, validation, and test sets and save the results as
   plaintext files.
4. Prepare the plaintext files by converting all symbols to integers and saving
   them in .pt files.

* `submit_generate_data_jobs.bash`: Submit jobs to sample all languages,
  generate datasets from them, and prepare them.
* `submit_generate_more_training_data_jobs.bash`: Submit jobs to generate a
  larger training set for a subset of the languages.

### Training Neural Networks

Next, we train neural networks on the datasets generated for each language.
These scripts are under `experiments/training/`.

* `submit_train_and_evaluate_jobs.bash`: Submit jobs to train and evaluate all
  models on all languages.
* `submit_continue_train_and_evaluate_jobs.bash`: Submit jobs to continue
  training and evaluation runs that were interrupted.
* `submit_train_and_evaluate_more_training_data_jobs.bash`: Submit jobs to train
  and evaluate models on larger training sets for a subset of the languages.
* `submit_continue_train_and_evaluate_more_training_data_jobs.bash`: Submit jobs
  to continue training and evaluation runs on more training data that were
  interrupted.

### Analysis

The relevant scripts are under `experiments/analysis/`. They should be run after
models are trained and evaluated.

* `submit_baseline_jobs.bash`: Submit jobs to calculate the baseline accuracy
  for each language.
* `submit_sensitivity_and_density_jobs.bash`: Submit jobs to estimate the
  sensitivity and density of all languages.
* `submit_class_summary_tables.bash`: Print the tables that show mean test
  accuracy and perfect rates for each architecture and language class.
* `print_random_bubble_diagram.bash`: Generate the Venn diagrams that summarize
  the results as `.tex` code.
* `print_csv.bash`: Generate a .csv file summarizing the results of all runs to
  be used by the linear mixed effects model.
* `run_linear_model.bash`: Run the linear mixed effects model on the results.
* `print_training_data_size_table.bash`: Print the tables that show mean test
  accuracy and perfect rates for different training data sizes.
