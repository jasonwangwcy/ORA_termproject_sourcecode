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

import HiGHS
main_opt() = HiGHS.Optimizer()
bilevel_opt() = main_opt()
simulation_opt() = optimizer_with_attributes(
    HiGHS.Optimizer,
    "threads" => 1, # single thread
    "presolve" => "off", # no presolve
    "solver" => "simplex", # dual simplex
    "simplex_strategy" => 1,
)
optim_opt() = optimizer_with_attributes(
    HiGHS.Optimizer,
    "threads" => 1, # single thread
    "presolve" => "off", # no presolve
    "solver" => "simplex", # dual simplex
    "simplex_strategy" => 1,
)
const DEFAULT_PARAMETERS = [
    "presolve" => "off", # no presolve
    "solver" => "simplex", # dual simplex
]
const RETRY_PARAMETERS = [
    [
        "presolve" => "on",
        "solver" => "ipm", # barrier method
    ],
    [
        "presolve" => "on",
        "solver" => "simplex", # primal simplex
        "simplex_strategy" => 4,
    ],
]