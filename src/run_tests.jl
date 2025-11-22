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

Test function that performs in-sample training and out-of-sample tests.
=#
function run_tests(
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
    sample_current,
    train_dataset_size_current,
)

    # Load case into `case` variable (uses "ps_parser.jl")
    case_path = joinpath(@__DIR__, "..", "data", CASE_NAME * ".m")
    if isfile(case_path)
        case = read_matlab_file(case_path)
    else
        error("File not found")
    end

    # Converts Matpower data and adds relevant information
    # (converter is at "ps_converter.jl")
    # Time series data is also created at this point
    # (see "ps_ts.jl")
    pd, ts = convert_matpower(
        case,
        N_LAGS,
        N_DEMANDS,
        N_ZONES,
        TEST_SIZE,
        DEFF_COEF,
        SPILL_COEF,
        COEF_VARIATION,
        DEMAND_FILE,
        train_dataset_size_current,
        sample_current,
    )

    # Send system data to cores
    @everywhere pd = $pd
    @everywhere ts = $ts

    # real AR parameters
    θd_real = ts.θd_real

    # Log to file
    println(FILE)
    println(FILE, "θd_real")
    println(FILE, raw(θd_real))

    @everywhere GC.gc(true)

    #=
        1.1 - Optimization of Least Squares (LS) model
    =#

    @everywhere GC.gc(true)
    time_run = time()

    # exogenous version
    θd_ls = multi_ls(ts.d_train, pd.n_demand_lags, pd.n_demand)
    forecast_error = ls_noise_var(ts.d_train, pd.n_demand_lags, pd.n_demand)
    reserve_base_value = 1.96 * forecast_error

    θru_st = OffsetArray(
        fill(reserve_base_value, pd.n_zones, 1 + pd.n_reserve_lags),
        1:pd.n_zones,
        0:pd.n_reserve_lags,
    )
    θrd_st = deepcopy(θru_st)
    for r = 1:pd.n_zones, p = 1:pd.n_reserve_lags
        θru_st[r, p] = 0.0
        θrd_st[r, p] = 0.0
    end

    # distribute reserve among zones
    if pd.n_zones > 1
        frac = zone_frac(pd)
        reserve_base_value
        for r = 1:pd.n_zones
            θru_st[r, 0] = reserve_base_value * frac[r]
            θrd_st[r, 0] = reserve_base_value * frac[r]
        end
    end

    println(FILE)
    println(FILE, "opt_ls_ex_time: $(time()-time_run)")

    #=
        1.2 - Simulations with LS model
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Simulate Least Squares (LS) in Test Data")
    obj_ls_ex_ts = simulate(pd, θd_ls, θru_st, θrd_st, ts.d_test, SIM_SLICES, "_ls_test")
    println("Simulate Least Squares (LS) in Train Data")
    obj_ls_ex_tr = simulate(pd, θd_ls, θru_st, θrd_st, ts.d_train, SIM_SLICES, "_ls_train")
    save_solution(local_dict[], "ls_ex", θd_ls, θru_st, θrd_st, obj_ls_ex_tr, obj_ls_ex_ts)
    println(FILE, "sim_ls_ex_time: $(time()-time_run)")
    println(FILE, "θd_ls, θru_st, θrd_st")
    println(FILE, raw(θd_ls), raw(θru_st), raw(θrd_st))
    flush(FILE)

    #=
        2.1 - Optimization with the Partial Application Driven model (LS-OPT)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Optimize Partial Application Driven (LS-OPT)")
    if EXACT_BILEVEL
        θd_ad_ls, θru_ad_ls, θrd_ad_ls, obj_raw_ad_ls =
            bilevel(pd, SOLVE_TIME_LIMIT, ts.d_train, in_θd = θd_ls)
    else
        obj_raw_ad_ls = NaN
        θd_ad_ls, θru_ad_ls, θrd_ad_ls = optim_solve(
            pd,
            SOLVE_TIME_LIMIT,
            ts.d_train,
            in_θd = copy(θd_ls),
            start_θru = copy(θru_st),
            start_θrd = copy(θrd_st),
        )
    end
    println(FILE)
    println(FILE, "opt_ls_op_time: $(time()-time_run)")


    #=
        2.2 - Simulation with the Partial Application Driven model (LS-OPT)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Simulate Partial Application Driven (LS-OPT) in Test Data")
    obj_ls_ad_ts =
        simulate(pd, θd_ad_ls, θru_ad_ls, θrd_ad_ls, ts.d_test, SIM_SLICES, "_ad_ls_test")
    println("Simulate Partial Application Driven (LS-OPT) in Train Data")
    obj_ls_ad_tr =
        simulate(pd, θd_ad_ls, θru_ad_ls, θrd_ad_ls, ts.d_train, SIM_SLICES, "_ad_ls_train")
    save_solution(
        local_dict[],
        "ls_op",
        θd_ad_ls,
        θru_ad_ls,
        θrd_ad_ls,
        obj_ls_ad_tr,
        obj_ls_ad_ts,
    )
    println(FILE, "sim_ls_op_time: $(time()-time_run)")
    println(FILE, "θd_ad_ls, θru_ad_ls, θrd_ad_ls")
    println(FILE, raw(θd_ad_ls), raw(θru_ad_ls), raw(θrd_ad_ls))
    flush(FILE)

    #=
    #=
        3.1 - Optimization with the Partial Application Driven model (OPT-EX)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Optimize Partial Application Driven (OPT-EX)")
    if EXACT_BILEVEL
        θd_ad_st, θru_ad_st, θrd_ad_st, obj_raw_ad_st =
            bilevel(pd, SOLVE_TIME_LIMIT, ts.d_train, in_θru = θru_st, in_θrd = θrd_st)
    else
        obj_raw_ad_st = NaN
        θd_ad_st, θru_ad_st, θrd_ad_st = optim_solve(
            pd,
            SOLVE_TIME_LIMIT,
            ts.d_train;
            start_θd = copy(θd_ls),
            in_θru = copy(θru_st),
            in_θrd = copy(θrd_st),
        )
    end
    println(FILE)
    println(FILE, "op_ex_time: $(time()-time_run)")

    #=
        3.2 - Simulation with the Partial Application Driven model (OPT-EX)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Simulate Partial Application Driven (OPT-EX) in Test Data")
    obj_ad_ex_ts =
        simulate(pd, θd_ad_st, θru_ad_st, θrd_ad_st, ts.d_test, SIM_SLICES, "_ad_st_test")
    println("Simulate Partial Application Driven (OPT-EX) in Train Data")
    obj_ad_ex_tr =
        simulate(pd, θd_ad_st, θru_ad_st, θrd_ad_st, ts.d_train, SIM_SLICES, "_ad_st_train")
    save_solution(
        local_dict[],
        "op_ex",
        θd_ad_st,
        θru_ad_st,
        θrd_ad_st,
        obj_ad_ex_tr,
        obj_ad_ex_ts,
    )
    println(FILE, "op_ex_time: $(time()-time_run)")
    flush(FILE)
    =#


    #=
        4.1 - Optimization with the Full Application Driven model (OPT-OPT)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Optimize Full Application Driven (OPT-OPT)")
    if EXACT_BILEVEL
        θd_ad, θru_ad, θrd_ad, obj_raw_ad_ad = bilevel(pd, SOLVE_TIME_LIMIT, ts.d_train)
    else
        obj_raw_ad_ad = NaN
        θd_ad, θru_ad, θrd_ad = optim_solve(
            pd,
            SOLVE_TIME_LIMIT,
            ts.d_train;
            start_θd = copy(θd_ls),
            start_θru = copy(θru_st),
            start_θrd = copy(θrd_st),
        )
    end
    println(FILE)
    println(FILE, "opt_op_op_time: $(time()-time_run)")

    #=
        4.2 - Simulation with the Full Application Driven model (OPT-OPT)
    =#

    @everywhere GC.gc(true)
    time_run = time()
    println("Simulate Full Application Driven (OPT-OPT) in Test Data")
    obj_ad_ad_ts = simulate(pd, θd_ad, θru_ad, θrd_ad, ts.d_test, SIM_SLICES, "_ats.d_test")
    println("Simulate Full Application Driven (OPT-OPT) in Train Data")
    obj_ad_ad_tr =
        simulate(pd, θd_ad, θru_ad, θrd_ad, ts.d_train, SIM_SLICES, "_ats.d_train")
    save_solution(local_dict[], "op_op", θd_ad, θru_ad, θrd_ad, obj_ad_ad_tr, obj_ad_ad_ts)
    println(FILE, "sim_op_op_time: $(time()-time_run)")
    println(FILE, "θd_ad, θru_ad, θrd_ad")
    println(FILE, raw(θd_ad), raw(θru_ad), raw(θrd_ad))
    flush(FILE)

    #=
        Print results in tabel form
    =#

    println(FILE)
    println(FILE, Dates.now())
    println(FILE)
    println(FILE, "demand    | reserve   | train      | test       ")
    println(FILE, "----------|-----------|------------|------------")
    Printf.@printf(
        FILE,
        "LS        | exogenous | %10.4f | %10.4f\n",
        obj_ls_ex_tr,
        obj_ls_ex_ts
    )
    # Printf.@printf(FILE, "opt       | exogenous | %10.4f | %10.4f\n", obj_ad_ex_tr, obj_ad_ex_ts)
    Printf.@printf(
        FILE,
        "LS        | opt       | %10.4f | %10.4f\n",
        obj_ls_ad_tr,
        obj_ls_ad_ts
    )
    if !isnan(obj_raw_ad_ls)
        Printf.@printf(FILE, "Bilevel Objetive      | %10.4f | -\n", obj_raw_ad_ls)
    end
    Printf.@printf(
        FILE,
        "opt       | opt       | %10.4f | %10.4f\n",
        obj_ad_ad_tr,
        obj_ad_ad_ts
    )
    if !isnan(obj_raw_ad_ad)
        Printf.@printf(FILE, "Bilevel Objetive      | %10.4f | -\n", obj_raw_ad_ad)
    end
    flush(FILE)

    #=
        Analysis for Linearly biased forecasts the benchmark model
    =#

    time_run = time()
    obj_ls_ex_tr_vec = Float64[]
    obj_ls_ex_ts_vec = Float64[]
    for reg in LINEAR_BIAS
        @everywhere GC.gc(true)

        println("sim_ls_test_$(reg)")
        obj_ls_ex_ts =
            simulate(pd, reg .* θd_ls, θru_st, θrd_st, ts.d_test, SIM_SLICES, "_ls_test")
        println("sim_ls_train_$(reg)")
        obj_ls_ex_tr =
            simulate(pd, reg .* θd_ls, θru_st, θrd_st, ts.d_train, SIM_SLICES, "_ls_train")
        save_solution(
            local_dict[],
            "ls_ex_$(reg)",
            θd_ls,
            θru_st,
            θrd_st,
            obj_ls_ex_tr,
            obj_ls_ex_ts,
        )

        Printf.@printf(
            FILE,
            "LS %4.4f | exogenous | %10.4f | %10.4f\n",
            reg,
            obj_ls_ex_tr,
            obj_ls_ex_ts
        )
        push!(obj_ls_ex_tr_vec, obj_ls_ex_tr)
        push!(obj_ls_ex_ts_vec, obj_ls_ex_ts)

        flush(FILE)
        flush(stdout)
    end

    println(FILE)
    println(FILE, "ls_reg_all_time: $(time()-time_run)")
    println(FILE)
    flush(FILE)

    println()
    println("demand    | reserve   | train      | test       ")
    println("----------|-----------|------------|------------")
    Printf.@printf("LS        | exogenous | %10.4f | %10.4f\n", obj_ls_ex_tr, obj_ls_ex_ts)
    Printf.@printf("LS        | opt       | %10.4f | %10.4f\n", obj_ls_ad_tr, obj_ls_ad_ts)
    if !isnan(obj_raw_ad_ls)
        Printf.@printf("Bilevel Objetive      | %10.4f | -\n", obj_raw_ad_ls)
    end
    Printf.@printf("opt       | opt       | %10.4f | %10.4f\n", obj_ad_ad_tr, obj_ad_ad_ts)
    if !isnan(obj_raw_ad_ad)
        Printf.@printf("Bilevel Objetive      | %10.4f | -\n", obj_raw_ad_ad)
    end
    for (i, reg) in enumerate(LINEAR_BIAS)
        Printf.@printf(
            "LS %4.4f | exogenous | %10.4f | %10.4f\n",
            reg,
            obj_ls_ex_tr_vec[i],
            obj_ls_ex_ts_vec[i],
        )
    end
    println()

    flush(stdout)

end
