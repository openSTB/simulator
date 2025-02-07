# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: CC0-1.0

def initialize(
    interface: str | None = None,
    nthreads: int = 1,
    local_directory: str | None = None,
    memory_limit: int | float | str = "auto",
    nanny: bool = False,
    dashboard: bool = True,
    dashboard_address: str = ":8787",
    protocol: str | None = None,
    worker_class: str = "distributed.Worker",
    worker_options: dict | None = None,
    comm: None = None,
    exit: bool = True,
) -> bool | None: ...
def send_close_signal(): ...
