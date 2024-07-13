# TraceWeaver

TraceWeaver is a research prototype for transparently tracing requests through a microservice without application implementation. This repository serves as the artifact for the associated academic paper ["TraceWeaver: Distributed Request Tracing for Microservices Without Application Modification"](https://sachin.cs.illinois.edu/papers/traceweaver-ashok-sigcomm24.pdf) which will appear in SIGCOMM'24.

# Setup and Organization

To clone this repository, run:

```git clone https://github.com/Sachin-A/TraceWeaver.git```

```src:``` The main traceweaver algorithm as well as baselines are provided within src.

```data:``` For convenience, trace data collected from each application is made available within the data directory. The trace data is converted to standard JSON format as supported by the Jaeger tracing framework.

```exps:``` The experiments folder contain one sub-directory per experiment mentioned in the paper. The provided bash scripts execute the experiments mentioned in the paper from scratch using traces from ```data/```.

```third_party:``` The applications used for evaluation are provided as git submodules in the third_party folder. Please follow instructions within the corresponding submodules to run the apps.

```utils:``` Contains miscelleanous utility scripts.

# Gurobi License

In order to use the Gurobi Solver used by TraceWeaver, a Gurobi Academic license is required. Please follow instructions at this link to set up your Gurobi license and environment variables before exercising the code: [Academic Named-User License](https://www.gurobi.com/features/academic-named-user-license/).

# Getting Involved

- Please reach out via email (sachina3@illinois.edu) for support.
- Send us a PR if you are interested in contributing code.
