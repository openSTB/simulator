---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Circular trajectory

**Plugin name**: `circular`
<br>
**Implementation**: [`openstb.simulator.system.trajectory.Circular`][]

This plugin provides an ideal circular trajectory, that is, one which moves around a
perfect circle at a constant speed and constant depth.


## Parameters

This plugin has four required parameters:

* `centre`: the position of the centre of the circle in global coordinates. This must be
  given as a three-element array.

* `radius`: the radius of the circle in metres.

* `speed`: how fast the system is moving in metres per second.

* `clockwise`: a Boolean flag which should be set to true for a clockwise circle and
  false for a counter-clockwise circle.

For example, a circle of 50m radius around the origin at 21m depth:

```toml
[trajectory]
plugin = "circular"
centre = [0, 0, 21]
radius = 50
speed = 2
clockwise = true
```


### Optional parameters

Two optional parameters can be used to further customise the circles followed by the
system:

* `num_circles`: a positive floating-point value giving the number of circles the system
  performs. This does not have to be a whole number; for example, a value of 1.5 means
  the system performs a full circle followed by a half circle, stopping opposite its
  starting point. This defaults to 1.

* `start_angle`: the angle in degrees from the centre of the circle to the start of the
  trajectory. An angle of 0 corresponds to the vector between circle centre and system
  being parallel to the x axis, and an angle of 90 to that vector being parallel to the
  y axis. This defaults to 0.


### Start time

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
plugin = "circular"
centre = [0, 0, 21]
radius = 50
speed = 2
clockwise = true
start_time = "2026-04-08 10:30:00"
```

[ISO 8601]: https://en.wikipedia.org/wiki/ISO_8601
[Unix time]: https://en.wikipedia.org/wiki/Unix_time

