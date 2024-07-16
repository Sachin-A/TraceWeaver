# TraceWeaver

TraceWeaver is a research prototype for transparently tracing requests through a microservice without application implementation. This repository serves as the artifact for the associated academic paper ["TraceWeaver: Distributed Request Tracing for Microservices Without Application Modification"](https://sachin.cs.illinois.edu/papers/traceweaver-ashok-sigcomm24.pdf) which will appear in SIGCOMM'24.

# Organization

The project is organized as follows:

```src/``` The main traceweaver algorithm as well as baselines are provided within ```src/```.

```data/``` For convenience, trace data collected from each application is made available within the ```data/``` directory. The trace data is converted to standard JSON format as supported by the Jaeger tracing framework.

```exps/``` The experiments folder contain one sub-directory per experiment mentioned in the paper. The provided bash scripts execute the experiments mentioned in the paper from scratch using traces from ```data/```.

```third_party/``` The applications used for evaluation are provided as git submodules in the ```third_party/``` folder. Please follow instructions within the corresponding submodules to run the apps.

```utils/``` Contains miscelleanous utility scripts.

# Setup

### Installation

To clone this repository, run:

	git clone https://github.com/Sachin-A/TraceWeaver.git && cd TraceWeaver

Install python (3.11.9) by running your platform equivalent of the following command:

    sudo apt-get update
    sudo apt-get install python3.11

Install virtualenv to create and activate a virtual environment *env* to isolate the project dependencies from your system packages:

    pip install virtualenv
    python3.11 -m venv env
    source env/bin/activate

Now install all dependencies within the virtual environment:

	pip install -r requirements.txt

### Gurobi License

In order to use the Gurobi Solver invoked by TraceWeaver, a Gurobi Academic license is required. Please follow instructions at this link ([Academic Named-User License](https://www.gurobi.com/features/academic-named-user-license/)) to set up your Gurobi license and environment variables before exercising the code.


# Status
Note that this artifact is still being updated. We are making the source code available now, but some functionality is being updated from our private development. In addition, we plan to provide more detailed instructions to run the benchmarks, some of which relies on large preprocessed datasets from the Alibaba cluster.

# Getting Involved

- Please reach out via email (sachina3@illinois.edu) for support.
- Send us a PR if you are interested in contributing code.
