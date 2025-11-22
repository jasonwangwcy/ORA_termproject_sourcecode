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

import Xpress
bilevel_opt() = Xpress.Optimizer()
simulation_opt() = optimizer_with_attributes(
    Xpress.Optimizer,
    "THREADS" => 1, # single thread
    "PRESOLVE" => 0, # no presolve
    "DEFAULTALG" => 2, # dual simplex
)
optim_opt() = optimizer_with_attributes(
    Xpress.Optimizer,
    "THREADS" => 1, # single thread
    "PRESOLVE" => 0, # no presolve
    "DEFAULTALG" => 2, # dual simplex
    "KEEPBASIS" => 1, # re-use basis on re-optimize
)
const DEFAULT_PARAMETERS = [
    "PRESOLVE" => 0, # no presolve
    "DEFAULTALG" => 2, # dual simplex
    "KEEPBASIS" => 1, # re-use basis on re-optimize
]
const RETRY_PARAMETERS = [
    [
        "KEEPBASIS" => 0, # ignore basis on re-optimize
    ],
    [
        "PRESOLVE" => 1,
        "DEFAULTALG" => 4, # barrier method
        "KEEPBASIS" => 0, # ignore basis on re-optimize
    ],
    [
        "PRESOLVE" => 1,
        "DEFAULTALG" => 3, # primal simplex
        "KEEPBASIS" => 0, # ignore basis on re-optimize
    ],
]