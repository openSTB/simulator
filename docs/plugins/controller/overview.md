---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Controller plugin

A controller is responsible for managing the overall simulation. It specifies the
configuration needed to describe a simulation, and provides a method to run such
simulations when given an instance of this configuration.

The controller will typically need to break the simulation down into tasks which utilise
other plugins specified by the configuration. It can then submit these tasks to a
cluster, monitor their status and then combine and store the results of the tasks when
complete.
