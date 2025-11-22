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

Simple helper functions.
=#

if VERSION < v"1.1.0"
    isnothing(x) = x === nothing
end
raw(arr::Vector{Float64}) = arr
raw(arr::Matrix{Float64}) = arr
raw(arr::T) where {T<:OffsetArray} = arr.parent
raw(arr::T) where {T<:JuMP.Containers.DenseAxisArray} = arr.data
raw(m::JuMP.Model) = m.moi_backend.optimizer.model.inner

JuMP.value(a::T) where {T<:Number} = a

function save_solution(dict, name, θd, θru, θrd, obj_train, obj_test)
    inner = Dict(
        "θd" => θd,
        "θru" => θru,
        "θrd" => θrd,
        "obj_train" => obj_train,
        "obj_test" => obj_test,
    )
    dict[name] = inner
end

typedict(x::T) where {T} = Dict(fn => typedict(getfield(x, fn)) for fn ∈ fieldnames(T))
typedict(x::Real) = x
typedict(x::String) = x
typedict(x::Function) = nothing
typedict(x::Array{T,N}) where {T<:Real,N} = x
typedict(x::OffsetArray{T,N,A}) where {T<:Real,N,A<:Array{T,N}} = x
