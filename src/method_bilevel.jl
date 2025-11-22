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

Exact solution method implementation with BilevelJuMP.
=#

function bilevel(
    pd::PhysicalData,
    time_limit,
    d;
    in_θd = nothing,
    in_θru = nothing,
    in_θrd = nothing,
)

    @unpack_PhysicalData pd

    AR_BND = 2 * maximum(d)

    m = BilevelModel()

    if isnothing(in_θd)
        @variable(Upper(m), -AR_BND <= θd[1:n_demand, 0:n_demand_lags] <= AR_BND)
    else
        θd = in_θd
    end

    if isnothing(in_θru)
        RU_BND = 1.1 * sum(pd.max_r_up)
        @variable(
            Upper(m),
            (n_reserve_lags == 0 ? 0.0 : -AR_BND) <=
            θru[1:n_zones, 0:n_reserve_lags] <=
            RU_BND
        )
    else
        θru = in_θru
    end

    if isnothing(in_θrd)
        RD_BND = 1.1 * sum(pd.max_r_dn)
        @variable(
            Upper(m),
            (n_reserve_lags == 0 ? 0.0 : -AR_BND) <=
            θrd[1:n_zones, 0:n_reserve_lags] <=
            RD_BND
        )
    else
        θrd = in_θrd
    end

    T = lastindex(d, 2)
    lm = NotBatchedModel(Lower(m), T)
    um = NotBatchedModel(Upper(m), T)

    g, ru, rd = build_schedule(pd, lm, θd, θru, θrd, d)

    build_dispatch(pd, um, g, ru, rd, d)

    BilevelJuMP.set_mode(m, BilevelJuMP.SOS1Mode())
    JuMP.set_optimizer(m, bilevel_opt)

    # JuMP.set_silent(m)
    if time_limit > 0
        JuMP.set_time_limit_sec(m, time_limit)
    end

    optimize!(m)

    stat = termination_status(m)
    if stat != MOI.OPTIMAL
        @warn "termination status was $(stat)"
        @warn "primal status was $(primal_status(m))"
        println(FILE, "termination status was $(stat)")
    end

    if primal_status(m) == MOI.FEASIBLE_POINT
        _θd = value.(θd)
        _θru = value.(θru)
        _θrd = value.(θrd)
    else
        println(FILE, "no feasible point found")
        _θd = fill!(similar(θd, Float64), 0.0)
        _θru = fill!(similar(θru, Float64), 0.0)
        _θrd = fill!(similar(θrd, Float64), 0.0)
    end

    return _θd, _θru, _θrd, JuMP.objective_value(m) / lastindex(d, 2)
end
