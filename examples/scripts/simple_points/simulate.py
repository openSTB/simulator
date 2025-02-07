# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import sys
from typing import Literal

import numpy as np
import quaternionic

from openstb.simulator.plugin import loader
from openstb.simulator.simulation.points import PointSimulation, PointSimulationConfig

# The local Dask cluster uses the multiprocessing module. This will import this
# script at the start of each worker process. If the code to configure and start the
# simulation is run during the import, this will lead to each worker trying to start
# another cluster and simulation, and so on. Instead, we put our simulation setup and
# execution inside a function, and use an if __name__ == "__main__" guard at the end of
# the script to only call it from the top-level execution; the workers will have a
# different __name__ value. This is the standard way of using multiprocessing and
# therefefore Dask local clusters.


def simulate(cluster: Literal["local"] | Literal["mpi"]):
    # Begin our configuration dictionary.
    config: PointSimulationConfig = {}

    # Each plugin is defined through a plugin specification dictionary. This takes the
    # name the plugin is registered under (see pyproject.toml for a list of the included
    # plugins) and a dictionary of any parameters it requires.
    #
    # If you want to use a custom plugin, you can use either of the following forms for
    # the name:
    #
    #     ClassName:package.module
    #
    #     ClassName:/path/to/file.py

    if cluster == "local":
        # Create a cluster on the local machine with 8 workers able to use up to ~40% of
        # the total memory (note that the memory is enforced on a best-effort basis).
        config["dask_cluster"] = loader.dask_cluster(
            {
                "name": "local",
                "parameters": {
                    "workers": 8,
                    "total_memory": 0.4,
                    "dashboard_address": ":8787",
                },
            }
        )

    elif cluster == "mpi":
        # Cluster is managed by MPI.
        config["dask_cluster"] = loader.dask_cluster(
            {
                "name": "mpi",
                "parameters": {
                    "dashboard_address": ":8787",
                },
            }
        )

    else:
        raise ValueError(f"Unknown cluster type '{cluster}'")

    # Initialise the cluster. The simulation method should also do this, but we don't
    # want to wait. In an MPI situation, each worker gets called with the same command
    # and so will reach this function. The initialise() method is what lets Dask take
    # control of the workers, so if we wait until the simulation starts each worker will
    # parse the configuration (including reading data off the disk, generating the
    # targets etc). For a local cluster, this function will not be reached by the
    # workers, but it does no harm to initialise the cluster here.
    config["dask_cluster"].initialise()

    # Use a 10m linear trajectory along the x axis at 1.5m/s.
    config["trajectory"] = loader.trajectory(
        {
            "name": "linear",
            "parameters": {
                "start_position": [0, 0, 0],
                "end_position": [10, 0, 0],
                "speed": 1.5,
            },
        }
    )

    # Decide when the sonar will transmit pings. Here, we ping at a constant interval of
    # 0.2s, starting at t=0 (the start of the trajectory) and with no ping closer than
    # 0.5s to the end of the trajectory.
    config["ping_times"] = loader.ping_times(
        {
            "name": "constant_interval",
            "parameters": {
                "interval": 0.2,
                "start_delay": 0,
                "end_delay": 0.5,
            },
        }
    )

    # The environment is spatially and temporally invariant.
    config["environment"] = loader.environment(
        {
            "name": "invariant",
            "parameters": {
                "salinity": 14.5,
                "sound_speed": 1480.0,
                "temperature": 11.2,
            },
        }
    )

    # Include two collections of point targets. The first is a rectangle with the given
    # size, position and normal ([0, 0, -1] points up -- remember the z axis is down)
    # filled with randomly placed points at a density of 10 per m^2. The reflectivity is
    # the fraction of incident amplitude that is scattered back to the sonar. The second
    # target is a single point at a given position.
    config["targets"] = [
        loader.point_targets(
            {
                "name": "random_point_rectangle",
                "parameters": {
                    "seed": 10671,
                    "Dx": 5,
                    "Dy": 120,
                    "centre": (5, 75, 10),
                    "normal": (0, 0, -1),
                    "point_density": 10,
                    "reflectivity": 0.06,
                },
            }
        ),
        loader.point_targets(
            {
                "name": "single_point",
                "parameters": {
                    "position": (5, 40, 10),
                    "reflectivity": 10,
                },
            }
        ),
    ]

    # Use the stop-and-hop approximation when calculating the travel time of the pulse.
    config["travel_time"] = loader.travel_time(
        {
            "name": "stop_and_hop",
            "parameters": {},
        }
    )

    # Apply two distortions: spherical spreading (1/r scaling to the amplitude on
    # each direction) and acoustic attenuation.
    config["distortion"] = [
        loader.distortion(
            {
                "name": "geometric_spreading",
                "parameters": {
                    "power": 1.0,
                },
            }
        ),
        loader.distortion(
            {
                "name": "anslie_mccolm_attenuation",
                "parameters": {
                    "frequency": "centre",
                },
            }
        ),
    ]

    # Define the signal the sonar will transmit; a Tukey-windowed LFM upchirp here.
    signal = loader.signal(
        {
            "name": "lfm_chirp",
            "parameters": {
                "f_start": 100e3,
                "f_stop": 120e3,
                "duration": 0.015,
                "rms_spl": 190,
                "rms_after_window": True,
                "window": {
                    "name": "tukey",
                    "parameters": {"alpha": 0.2},
                },
            },
        }
    )

    # Set the desired orientation of the transducers. Without rotation, the normal of
    # the transducer, i.e., the direction it is pointing, is [1, 0, 0] (x is forward, y
    # is starboard and z is down, so straight ahead). The function used here uses a
    # rotation vector to create a quaternion. The magnitude gives the angle of rotation
    # and the normalised version the axis. Here, we rotate 90 degrees about z (point to
    # starboard) and 15 degrees around x (15 degrees down).
    q_yaw = quaternionic.array.from_rotation_vector([0, 0, np.pi / 2])
    q_tilt = quaternionic.array.from_rotation_vector([np.radians(15), 0, 0])
    q_transducer = q_tilt * q_yaw

    # Define a common far-field beampattern for the transducers. Note that this is just
    # a distortion attached to the transducers; we could add this to the list of
    # distortion plugins above and not pass it to the transducers to achieve the same
    # result.
    beampattern = {
        "name": "rectangular_beampattern",
        "parameters": {
            "width": 0.015,
            "height": 0.03,
            "transmit": True,
            "receive": False,
            "frequency": "centre",
        },
    }

    # Define the transmitting transducer.
    transmitter = loader.transducer(
        {
            "name": "generic",
            "parameters": {
                "position": [0, 1.2, 0.3],
                "orientation": q_transducer,
                "beampattern": beampattern,
            },
        }
    )

    # And then the list of receiving transducers.
    beampattern["parameters"]["transmit"] = False
    beampattern["parameters"]["receive"] = True
    receivers = [
        loader.transducer(
            {
                "name": "generic",
                "parameters": {
                    "position": [x, 1.2, 0],
                    "orientation": q_transducer,
                    "beampattern": beampattern,
                },
            }
        )
        for x in [-0.1, -0.05, 0, 0.05, 0.1]
    ]

    # Combine all this into a System plugin. They could also be placed directly in the
    # configuration under the transmitter, receivers and signal keys.
    config["system"] = loader.system(
        {
            "name": "generic",
            "parameters": {
                "transmitter": transmitter,
                "receivers": receivers,
                "signal": signal,
            },
        }
    )

    # Internally, the simulation result is stored in a Zarr group. You could choose to
    # directly load the results from this format, or you could configure a conversion
    # plugin to write them to a different format. The following converts the result to
    # an uncompressed NumPy .npz file; this can then be loaded with
    # np.load("example_sim.py") which returns a mapping interface.
    config["result_converter"] = loader.result_converter(
        {
            "name": "numpy",
            "parameters": {
                "filename": "simple_points.npz",
                "compress": False,
            },
        }
    )

    # If you prefer, you could convert this to a MATLAB file instead. This uses the
    # `scipy.io.savemat` function provided by SciPy; note that this only supports "5"
    # (MATLAB 5 and up) and "4" as the format arguments, and not the newer HDF-backed
    # formats.
    # config["result_converter"] = loader.result_converter(
    #     {
    #         "name": "matlab",
    #         "parameters": {
    #             "filename": "simple_points.mat",
    #             "format": "5",
    #             "long_field_names": False,
    #             "do_compression": False,
    #             "oned_as": "row",
    #         },
    #     }
    # )

    # Initialise the simulator class. In the future, this is intended be a plugin once a
    # suitable interface has been determined. Note that the simulator will refuse to
    # overwrite an existing output file, so you will need to either delete it or change
    # the output name in the simulation definition if you want to re-run the simulation.
    # We manually specify how many targets to include in each chunk of work, and the
    # details about the system sampling. The output will be in the complex baseband.
    sim = PointSimulation(
        result_filename="simple_points.zarr",
        targets_per_chunk=1000,
        sample_rate=30e3,
        baseband_frequency=110e3,
    )

    # And finally, run the simulation. While it is running, you can access the Dask
    # dashboard at 127.0.0.1:8787 to see various diagnostic plots about how the cluster
    # is being utilised.
    sim.run(config)


if __name__ == "__main__":
    # First argument will be name of the script.
    nargs = len(sys.argv) - 1

    # Default to directly on the local machine.
    if nargs == 0:
        cluster = "local"

    # User specified.
    elif nargs == 1:
        cluster = sys.argv[1]
        if cluster not in {"local", "mpi"}:
            raise SystemExit(f"Usage: {sys.argv[0]} [local|mpi]")

    # Too many arguments for us to handle.
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [local|mpi]")

    simulate(cluster)
