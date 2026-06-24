---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Linear trajectory

**Plugin name**: `linear`
<br>
**Implementation**: [`openstb.simulator.system.trajectory.Linear`][]

This plugin provides an ideal linear trajectory, that is, one which moves from a start
point to an end point in a straight line at a constant speed.


## Parameters

This plugin has three required parameters:

* `start_position`: the position the trajectory starts in global coordinates. This must
  be given as a three-element array.

* `end_position`: the position the trajectory ends in global coordinates. This must be
  given as a three-element array.

* `speed`: how fast the system is moving in metres per second.

For example, a 100m long trajectory along the global x axis taking 50s:

```toml
[trajectory]
start_position = [0, 0, 0]
end_position = [100, 0, 0]
speed = 2
```

By default, the start time of the trajectory is set to the time that the plugin instance
is created. If you want to set a specific date and time as the start of the trajectory,
the optional `start_time` parameter can be set. This can be specified in several ways:

* A string in [ISO 8601][] format.

* An integer giving the number of seconds since midnight on 1 January 1970 ([Unix
  time][]).

* If initialising the plugin from within Python, a [datetime][datetime.datetime]
  instance.

Internally, the start times will be converted to UTC as required by the plugin
interface. If the input time has no timezone information, it will be assumed to already
be UTC. For example, if we wanted the previous trajectory to start at 10:30am UTC on 8
April 2026, we could configure it as follows.

```toml
[trajectory]
start_position = [0, 0, 0]
end_position = [100, 0, 0]
speed = 2
start_time = "2026-04-08 10:30:00"
```

[ISO 8601]: https://en.wikipedia.org/wiki/ISO_8601
[Unix time]: https://en.wikipedia.org/wiki/Unix_time
