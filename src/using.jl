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

External packages loader and solver basic defitions.
=#

# - optimization
using JuMP
import ParameterJuMP
import Optim
import BilevelJuMP: BilevelJuMP, BilevelModel, Upper, Lower
# - base
import Dates
import Random
import Statistics
# - structres
import OffsetArrays: OffsetArray
import Parameters
# - IO
import Printf
import JLD2
# - Parallel
import SharedArrays
import DelimitedFiles
