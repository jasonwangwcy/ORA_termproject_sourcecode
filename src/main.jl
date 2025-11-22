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

Main file.
This should be the only file loaded by julia by the user.
The other files are loaded by this one.
For configurations of tests the user is directed to:
configurations.jl
=#


println("""
        ######################################################################
        #
        #    Application Driven Learning code
        #
        ######################################################################
        #  Copyright 2024, Joaquim Dias Garcia, and contributors
        #  This Source Code Form is subject to the terms of the Mozilla Public
        #  License, v. 2.0. If a copy of the MPL was not distributed with this
        #  code, You can obtain one at https://mozilla.org/MPL/2.0/.
        ######################################################################
        # This code is part of the publication:
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
        """)
println("----")

# load user configurations

include("configuration.jl")

# --------

println("Selected options:")
@show N_CORES
@show CASE_NAME
@show DEMAND_FILE
@show EXACT_BILEVEL
@show SOLVER
@show TEST_SIZE
@show TRAIN_DATASET_SIZES
@show TRAIN_SAMPLES
@show N_LAGS
println("----")

# assert Julia version
if VERSION != v"1.8.1"
    error("Expected julia version 1.8.1, got $(VERSION)")
end

println("Initializing deps")

# Initialize base environment on main core
using Pkg
cd(joinpath(@__DIR__))
Pkg.activate(".")
Pkg.instantiate()
@static if SOLVER == "Gurobi"
    Pkg.add(name = "Gurobi", version = "0.11.5")
elseif SOLVER == "Xpress"
    Pkg.add(name = "Xpress", version = "0.15.3")
elseif SOLVER == "HiGHS"
    # already installed by default
else
    error("Unsupported solver")
end

println("Finished initializing deps")

# Initilize Distributed package in the main core (as others were not started yet)
using Distributed

# Initialize additional cores
addprocs(max(0, N_CORES - nprocs()))
# extra check necessary if the user has modified the script
@assert N_CORES == nprocs()
@assert maximum(workers()) == N_CORES

# Repeat base environment loading in all cores
@everywhere begin
    println("Activating process $(myid())")

    cd(joinpath(@__DIR__))
    using Pkg
    Pkg.activate(".")
    using Distributed

    println("Finished activating process $(myid())")
end

#=
    Load external dependencies
=#
# First load only on main core needed for precompilation
println("Including deps in main core")
include("using.jl")
@static if SOLVER == "Gurobi"
    include("solver_gurobi.jl")
elseif SOLVER == "Xpress"
    include("solver_xpress.jl")
elseif SOLVER == "HiGHS"
    include("solver_highs.jl")
end
sleep(1.0)

if N_CORES > 1
    println("Including deps in all worker cores")
    @everywhere workers() include("using.jl")
    @static if SOLVER == "Gurobi"
        @everywhere workers() include("solver_gurobi.jl")
    elseif SOLVER == "Xpress"
        @everywhere workers() include("solver_xpress.jl")
    elseif SOLVER == "HiGHS"
        @everywhere workers() include("solver_highs.jl")
    end
end

println("Finished including deps in all cores")

#=
    Load code files
=#
@everywhere begin # repeat in all processes:

    # helper definitions
    include("help.jl")

    # data generation and loading
    include("matpower_parser.jl")
    include("builder_time_series.jl")
    include("builder_system_data.jl")

    # problem and simulation definition
    include("optimization_models.jl")

    # solution methods
    include("ols.jl")
    include("batch_model.jl")
    include("method_heuristic.jl")
    include("method_bilevel.jl")

    include("run_tests.jl")

end # @everywhere

#=
Global variables used to:
- save results to file
- save final results
- hold temporary solutions during the optimization process 
=#

PATH = joinpath(@__DIR__, "..")
const FILE = open(joinpath(PATH, CASE_NAME * ".log"), "w")
local_dict = [Dict()]
main_dict = Dict()
main_dict["case"] = CASE_NAME

#=
Main loop
=#

for i in TRAIN_DATASET_SIZES, sample_ in TRAIN_SAMPLES
    local_dict[] = Dict()
    if !haskey(main_dict, "$i")
        main_dict["$i"] = [Dict() for i = 1:maximum(TRAIN_SAMPLES)]
    end
    main_dict["$i"][sample_] = local_dict[]
    println(FILE, "train size: $(i)")
    println(FILE, "train sample: $(sample_)")
    run_tests(
        CASE_NAME::String,
        N_LAGS::Int,
        N_DEMANDS::Int,
        N_ZONES::Int,
        TEST_SIZE::Int,
        DEFF_COEF::Float64,
        SPILL_COEF::Float64,
        COEF_VARIATION::Float64,
        DEMAND_FILE::String,
        SOLVE_TIME_LIMIT::Float64,
        SIM_SLICES,
        local_dict,
        sample_,
        i,
    )
    flush(FILE)
end
JLD2.save(joinpath(PATH, "$(CASE_NAME).jld2"), main_dict)

# close(FILE)
