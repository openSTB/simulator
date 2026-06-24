---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Environment plugin

The properties of the environment that the system is operating are provided by an
environment plugin. This includes the speed of sound, temperature and salinity.

!!! Note
    The environment plugin only provides the properties. To make use of varying
    environmental properties, the other plugins used by a simulation (e.g., distortion
    plugins implementing attenuation or travel time plugins) must include support for
    this.
