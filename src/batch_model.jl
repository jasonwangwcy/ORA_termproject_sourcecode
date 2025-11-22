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

Helper functions that define a batched optimizer:
- BatchedModel behaves like a JuMP Model for all effects.
But a block diagonal model can be effectivelly defined
without explicitly creating multiple models.
- NonBatchedModel is exactly the same as a JuMP Model
as the number of blocks is exactly one.
=#

struct BatchParameterRef{M}
    model::M
    ref::Int
end

struct BatchedModel{M}
    num_batches::Int
    num_t::Int
    models::Vector{M}
    params::Vector{Vector{ParameterJuMP.ParameterRef}}
end

# creates a batched model that behaves exactly as a non batched model
# in this case the `num_batches` = `1` meaning that there is a
# single batch os size `num_t`
function NotBatchedModel(model::M, num_t) where {M}
    return BatchedModel{M}(1, num_t, [model], Vector{ParameterJuMP.ParameterRef}[])
end

# create a batched model that contains one optimization subproblem
# for each `num_t`, hence, `num_batches = num_t`.
function BatchedModel(factory, num_t::Integer, with_param::Bool = false)
    num_batches = num_t
    if with_param
        vec_of_models = [direct_model(factory) for _ = 1:num_t]
        for m in vec_of_models
            ParameterJuMP.set_no_duals(m)
            ParameterJuMP.set_not_lazy_duals(m)
        end
    else
        vec_of_models = [direct_model(factory) for _ = 1:num_t]
    end
    return BatchedModel{eltype(vec_of_models)}(
        num_batches,
        num_t,
        vec_of_models,
        Vector{ParameterJuMP.ParameterRef}[],
    )
end

# gets the index of the model (batch) that contains the i-th element (stage)
function _index(bm::BatchedModel, i::Int)::Int
    if bm.num_batches == bm.num_t
        return i
    end
    return 1
end

# returns indices of all models
# each one is a batch possibly containing many subproblems
function batch_list(bm::BatchedModel)
    return ifelse(bm.num_batches == bm.num_t, 1:bm.num_t, 1:1)
end

# returns the single model representing batch i of subproblems
function model_from_batch(bm::BatchedModel{M}, i) where {M}
    return bm.models[i]::M
end

# gets all stages contained in batch i (the i-th model)
function stages(bm::BatchedModel, i::Integer)::UnitRange{Int}
    if !(1 <= i <= bm.num_batches)
        error("Batch $i was requested, but maximum number of batches is $(bm.num_batches).")
    end
    if bm.num_batches == bm.num_t
        return i:i
    end
    return 1:bm.num_t
end

# gets the parameter associated to model (batch) containing stage t
function batched(p::BatchParameterRef{M}, t) where {M}
    return p[t]
end

# fallback for the case the parameter is a variable in the model
function batched(q, t)
    return q
end

# gets model from stage t
function Base.getindex(bm::BatchedModel{M}, i::Int) where {M}
    if !(1 <= i <= bm.num_t)
        error("Stage $i was requested, but maximum number of stages is $(bm.num_t).")
    end
    j = _index(bm, i)
    return bm.models[j]::M
end

function JuMP.set_silent(bm::BatchedModel{M}) where {M}
    for m in bm.models
        set_silent(m)
    end
    return nothing
end

function JuMP.set_time_limit_sec(bm::BatchedModel{M}, val) where {M}
    for m in bm.models
        JuMP.set_time_limit_sec(m, val)
    end
    return nothing
end

function JuMP.optimize!(bm::BatchedModel{M}) where {M}
    for m in bm.models
        optimize!(m::M)
        if termination_status(m) != MOI.OPTIMAL
            for attrs in RETRY_PARAMETERS
                for j in attrs
                    set_optimizer_attribute(m, j[1], j[2])
                end
                optimize!(m::M)
                if termination_status(m) == MOI.OPTIMAL
                    break
                end
            end
            for (param, val) in DEFAULT_PARAMETERS
                set_optimizer_attribute(m, param, val)
            end
        end
    end
    return nothing
end

function model_ok(var)
    m = JuMP.owner_model(var)
    return JuMP.termination_status(m) == MOI.OPTIMAL
end

function JuMP.termination_status(bm::BatchedModel)
    c = 0
    for m in bm.models
        st = termination_status(m)
        if st != MOI.OPTIMAL
            c += 1
        end
    end
    if c != 0
        error("Model optimization failed $c times")
    end
    return MOI.OPTIMAL
end

function JuMP.objective_value(bm::BatchedModel)
    val = 0.0
    for m in bm.models
        val += JuMP.objective_value(m)
    end
    return val
end

# add the same parameter in all models
function ParameterJuMP._add_parameter(bm::BatchedModel, val::Float64)
    pars = [ParameterJuMP._add_parameter(m, val) for m in bm.models]
    push!(bm.params, pars)
    return BatchParameterRef{BatchedModel}(bm, length(bm.params))
end

function Base.getindex(p::BatchParameterRef{M}, i::Integer) where {M}
    bm = p.model
    if !(1 <= i <= bm.num_t)
        error("Stage $i was requested, but maximum number of stages is $(bm.num_t).")
    end
    j = _index(bm, i)
    return bm.params[p.ref][j]
end

function JuMP.set_value(p::BatchParameterRef{M}, val::Float64) where {M}
    bm = p.model::M
    p_vec = bm.params[p.ref::Int]::Vector{ParameterJuMP.ParameterRef}
    for pp in p_vec
        set_value(pp::ParameterJuMP.ParameterRef, val)
    end
    return nothing
end
