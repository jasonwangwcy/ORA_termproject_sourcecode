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

import Gurobi
main_opt() = Gurobi.Optimizer()
GUROBI_ENV = Gurobi.Env()
bilevel_opt() = main_opt()
simulation_opt() = optimizer_with_attributes(
    Gurobi.Optimizer,
    "Threads" => 1, # single thread
    "Presolve" => 0, # no presolve
    "Method" => 1, # dual simplex
)
optim_opt() = optimizer_with_attributes(
    Gurobi.Optimizer,
    "Threads" => 1, # single thread
    "Presolve" => 0, # no presolve
    "Method" => 1, # dual simplex
    "LPWarmStart" => 1, # re-use basis on re-optimize
)
const DEFAULT_PARAMETERS = [
    "Presolve" => 0, # no presolve
    "Method" => 1, # dual simplex
    "LPWarmStart" => 1, # re-use basis on re-optimize
]
const RETRY_PARAMETERS = [
    [
        "LPWarmStart" => 0, # ignore basis on re-optimize
    ],
    [
        "Presolve" => 1,
        "Method" => 2, # barrier method
        "LPWarmStart" => 0, # ignore basis on re-optimize
    ],
    [
        "Presolve" => 1,
        "Method" => 0, # primal simplex
        "LPWarmStart" => 0, # ignore basis on re-optimize
    ],
]