######################################################################
#  Copyright 2024, Joaquim Dias Garcia, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.
######################################################################
# This file is part of the publication:
#  Application-Driven Learning: A Closed-Loop
#  Prediction and Optimization Approach Applied to
#  Dynamic Reserves and Demand Forecasting
# by:
#  Joaquim Dias Garcia, Alexandre Street,
#  Tito Homem-de-Mello & Francisco D. Munoz
######################################################################
# This code base is part of an academic work and should not be
# considered ready for production.
######################################################################

#=
Info:

User configurations.
With the options available here the case studies from the paper can be executed.
This is the only file that should be modified by the user.
=#

#=
    Number of total cores in the optimizaiton process.
=#
# N_CORES = 1
# N_CORES = 2
# N_CORES = 4
 N_CORES = 8
# N_CORES = 64 # recommended for large cases

#=
    Case data
    - name of the case (matpower .m) files from "data" folder
=#
# CASE_NAME = "pglib_opf_case24_ieee_rts" # example
# CASE_NAME = "simple_case1" # (Secs 8.3, 8.4)
 CASE_NAME = "pglib_opf_case6468_rte" # (Sec 8.5)
# CASE_NAME = "pglib_opf_case6470_rte" # (Secs 8.5 and 8.6)
# CASE_NAME = "pglib_opf_case6495_rte" # (Sec 8.5)
# CASE_NAME = "pglib_opf_case6515_rte" # (Sec 8.5)
# CASE_NAME = "pglib_opf_case9241_pegase" # (Sec 8.5)
# CASE_NAME = "pglib_opf_case13659_pegase" # (Sec 8.5)

#=
    Demand data
    - if empty generates synthetic data (Secs 8.3, 8.4, 8.5)
    - else uses file (Sec 8.6)
=#
# DEMAND_FILE = joinpath(@__DIR__, "..", "data", "demand.csv") # (Sec 8.6)
DEMAND_FILE = "" # automatically generated load data (Secs 8.3, 8.4, 8.5)

#=
    Solution strategy
    - if true, exact mode is applied relying on BilevelJuMP to exactly solve bilevel model
        only small models can be solved (small systems with limited number of samples).
        only used in Sec 8.3
    - if false, heuristic method is applied. Recommended. (Secs 8.3, 8.4, 8.5, 8.6)
=#
# EXACT_BILEVEL = true
EXACT_BILEVEL = false # default

#=
    Solver for optimization subproblems
=#
# SOLVER = "HiGHS" # example
 SOLVER = "Gurobi" #
# SOLVER = "Xpress" #

#=
    Linear Bias for Linearly biased forecasts

    See alpha coefficient in section 7.

    This is only used in sec 8.6.
=#
# LINEAR_BIAS = Int[] # default
LINEAR_BIAS = [1.005, 1.01] # example
# LINEAR_BIAS = collect(1.0025:0.0025:1.05) # (Sec 8.6)

#=
    Out of sample dataset size / number os test samples
    - 7 * 24 = 168, was used in the final case study (Sec 8.6)
    - 10_000, was used in other cases (Secs 8.4, 8.5)
    Note: Sec 8.3 does not perform out of sample analysis
=#
TEST_SIZE = 7 * 24 # example
# TEST_SIZE = 10_000 # (Secs 8.4, 8.5)
# TEST_SIZE = 7 * 24 # (Sec 8.6)

#=
    Vector of traning dataset sizes (sample size)

    Multiple samples of datasets can be run in a single loop, hence, a vector.
=#
TRAIN_DATASET_SIZES = [1 * 168] # example
# TRAIN_DATASET_SIZES = [15, 25, 50, 75] # sec 8.3
# TRAIN_DATASET_SIZES = collect(50:50:1000) # sec 8.4
# TRAIN_DATASET_SIZES = [600] # Set 8.5
# TRAIN_DATASET_SIZES = [168] # Set 8.6

#=
    Number of samples for each dataset size

    For each dataset size (set above) multiple different datasets can be used to train
    and then be tested. We call each of these datasets of fixed size a dataset sample
    or a training sample (a sample of multiple records of data to train the model).
=#
TRAIN_SAMPLES = 1:5 # example

#=
    Number os lag in the load autorregressive model
=#
# N_LAGS = 1 # Secs 8.3, 8.4, 8.5
# N_LAGS = 24 # Sec 8.6
N_LAGS = ifelse(DEMAND_FILE == "", 1, 24) # automatic


#=
    Number of demand time series.

    If the the system has fewer buses, the number of buses is used instead.
=#
N_DEMANDS = 20 # default

#=
    Number of reserve zones

    If the the system has fewer buses, the number of buses is used instead.
    If set to zero, will read data from MATPOWER file
=#
N_ZONES = 10 # default
# N_ZONES = 0

#=
    System data fixed in all tests
=#

# Coefficient of variation of load: standard deviation divided by the average of load
COEF_VARIATION = 0.4 # default

# Deficit cost multiplier:
# how much the deficit cost is more expensive than the most expensive plant
DEFF_COEF = 8.0 # default

# Energy spillage cost multiplier:
# how much the energy spillage cost is more expensive than the most expensive plant
SPILL_COEF = 3.0 # default

#=
    Solution strategy additional parameters
=#

#=
    Solution method time limit.

    - If heuristic solution method (EXACT_BILEVEL = false):
    It is passed directly to the external function "Optim.optimize"
    - else
    Is is passed directly to the MIP solver by BilevelJuMP
=#
# SOLVE_TIME_LIMIT = 120.0 # example
 SOLVE_TIME_LIMIT = 1800.0 # default
# SOLVE_TIME_LIMIT = 7200.0 # exact bilevel


# number of parallel pieces in the final simulation
#=
    Out-of-sample simulation slices

    Number of slices to split the out-of-sample load sample
    this allows to perform the out-of sample analysis in parallel
    for smaller slices of the entire load data.
=#
SIM_SLICES = 3 * 64 # default
