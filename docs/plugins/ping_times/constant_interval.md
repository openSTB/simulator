---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Constant interval

**Plugin name**: `constant_interval`
<br>
**Implementation**: [`openstb.simulator.system.ping_times.ConstantInterval`][]

This plugin models a system operating with a constant ping repetition period. The
interval between pings is specified by the `interval` parameter, a floating-point value
in seconds. The delay between the trajectory starting and the first ping being
transmitted is given by the `start_delay` parameter, also a floating-point value in
seconds. The minimum delay between the final ping being transmitted and the end of the
trajectory is given by the `end_delay` parameter, another floating-point value in
seconds. All three parameters are required; the two delays can be set to zero if not
relevant.
