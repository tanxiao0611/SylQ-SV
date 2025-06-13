# SylQ-SV Artifact

This repository provides the scripts and configurations needed to reproduce the experiments described in paper, "SYLQ-SV: Scaling Symbolic Execution of Hardware Designs with Query Caching," using the SylQ-SV symbolic execution engine for Verilog/SystemVerilog RTL designs.

## Setup

Requirements
--------------------
* A Linux-based system (tested on Ubuntu)
* Recommended: 32+ GB RAM 

* Python3: 3.9 or later
* pySlang 7.0: `python3 -m pip install pyslang`
* Redis 7.4: 

    `sudo apt-get install lsb-release curl gpg` 

    `curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg`

    `sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg`

    `echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list`

    `sudo apt-get update`

* redis: run `python3 -m pip install redis`
* z3: run `python3 -m pip install z3-solver`
* Icarus Verilog: 10.1 or later: run `sudo apt install iverilog`
* Jinja 2.10 or later: run `python3 -m pip install jinja2`
* PLY 3.4 or later: run `python3 -m pip install ply`
* networkx: run `python3 -m pip install networkx`
* matplotlib: run `python3 -m pip install matplotlib`

### Directory Structure
```
├── designs/                # RTL benchmark designs (e.g., OR1200, Hack@DAC, OpenTitan)
├── results/                # Output from all experiments (auto-generated)
├── scripts/                # Individual bash scripts 
├── main.py                 # Entry point for SylQ-SV
├── cache_analysis.py       # Script to analyze Redis SMT query cache
├── Makefile                # Automates experiment runs
└── README.md               # You're here
```


## Getting Started

1. Install dependencies above.
2. Set up result directories:

```make init```

## Kick-the Tires

Goal: Run SylQ-SV on the OR1200 for 1 clock cycle for 5 minutes with query caching enabled, and inspect the number of symbolic paths and solver queries generated.

To run the experiment, execute the following command:

`python3 -m main 1 designs/benchmarks/buggy-or1200/or1200.F --use_cache --explore_time=300> out.txt`

The expected output for the kick-the-tires test is a summary of the number
of paths and branch points visited. The number of paths and branch points explored will vary based on your environment, but we saw around 50,000 paths exploredand 8,100 branch points explored. 

## Run experiments

1. 24-hour end-to-end symbolic exploration (w/ and w/o cache):

```make explore```

2. Assertion violation checks (for OR1200 and Hack@DAC SoCs):

```make assert-check```

3. Assertion checking with merge queries enabled:

```make merge-queries```

4. 1-hour symbolic execution comparison with and without caching:

```make cache-compare```

5. Analyze persistent Redis SMT query cache:

```make analyze-cache```

## Notes

* All experiments write results to the `results/<design>/<experiment_type>/out.txt` file.
* To run a single experiment manually, use:
`python3 -m main 1 designs/<design>/<top_module>.sv --use_cache=true > out.txt`
* For assertion checking, embed SVA assertions directly into the top-level module of your RTL design.
* The Redis cache file (`cache.rdb`) is persistent across runs and used to analyze query reuse.



