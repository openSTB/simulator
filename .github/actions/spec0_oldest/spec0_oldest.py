# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Find oldest versions of dependencies supported per SPEC0."""

from datetime import date, datetime
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version
import requests

# For core packages, SPEC0 says versions released within the last 2 years.
today = date.today()
cutoff = today.replace(year=today.year - 2)


def get_version(filename: str) -> Version | None:
    """Get version of sdist or wheel from its filename.

    This excludes pre-release versions.

    Parameters
    ----------
    filename
        The filename uploaded to PyPI.

    Returns
    -------
    Version | None
        The version of the filename, or None if it could not be determined.

    """
    if not filename:
        return None

    try:
        _, ver, _, _ = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        try:
            _, ver = parse_sdist_filename(filename)
        except InvalidSdistFilename:
            return None

    if ver.is_prerelease:
        return None

    return ver


def oldest_spec0_version(requirement: str) -> str:
    """Determine the oldest SPEC0 version for a requirement.

    If the oldest version has been yanked from PyPI, the oldest version within the SPEC0
    window which has not been yanked will be used.

    Parameters
    ----------
    requirement
        The dependency requirement as written in pyproject.toml, potentially including
        optional dependency sets and version limits.

    Returns
    -------
    str
        The specification to give to uv to install the appropriate version including
        single quotes, e.g., "'numpy==2.4.0'".

    """
    # Parse the requirement.
    rq = Requirement(requirement)

    # Load package information from PyPI.
    resp = requests.get(
        f"https://pypi.org/simple/{rq.name}",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    # Loop through all uploaded files.
    oldest_version = None
    oldest_yanked = False
    oldest_point = None
    for upload in data["files"]:
        if "upload-time" not in upload or "filename" not in upload:
            continue

        # Ignore releases before the cutoff.
        release_date = datetime.fromisoformat(upload["upload-time"]).date()
        if release_date < cutoff:
            continue

        # Ignore pre-releases and anything we can't parse the version of.
        version = get_version(upload["filename"])
        if not version:
            continue

        # Only include major or minor releases, apart from some special-cases we know
        # don't follow semver in this way.
        if version.micro > 0 and rq.name not in {"quaternionic"}:
            if oldest_point is None or version < oldest_point:
                oldest_point = version
            continue

        # Ensure this version matches any version limits in the requirement.
        if not rq.specifier.contains(version):
            continue

        # And track the oldest version we have found.
        if oldest_version is None or version < oldest_version:
            oldest_version = version
            oldest_yanked = upload.get("yanked", False)

    # Nothing matched our criteria. If there was no non-point release in the window, use
    # the oldest point release.
    if oldest_version is None:
        if oldest_point is not None:
            oldest_version = oldest_point
        else:
            raise ValueError(f"could not determine version for {rq.name}")

    # If the oldest release has been yanked, find the oldest version younger than it.
    if oldest_yanked:
        nearest = None

        for upload in data["files"]:
            # Ignore yanked versions and anything with a missing filename.
            if upload.get("yanked", False):
                continue
            if "filename" not in upload:
                continue

            # Get the version.
            version = get_version(upload["filename"])
            if not version:
                continue

            # Ignore anything older than the originally determined but yanked version.
            if version < oldest_version:
                continue

            # And track the oldest available version meeting these criteria.
            if nearest is None or version < nearest:
                nearest = version

        # Everything newer has been yanked.
        if nearest is None:
            raise ValueError(f"could not find non-yanked version for {rq.name}")

        oldest_version = nearest

    # Combine the version we have found with the package name and any optional
    # dependency sets to form the final spec.
    if rq.extras:
        return f"'{rq.name}[{','.join(rq.extras)}]=={oldest_version}'"
    return f"'{rq.name}=={oldest_version}'"


# Load the pyproject.toml file.
pp = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
with pp.open("rb") as f:
    pyproject = tomllib.load(f)

# And get the oldest SPEC0 for all main dependencies. We don't currently include the MPI
# optional dependencies here as the older versions tend to need to be compiled which we
# don't want to deal with at the moment.
specs = []
for dep in pyproject["project"]["dependencies"]:
    specs.append(oldest_spec0_version(dep))

# And print the list of specs in GitHub output format.
print(f"package-spec={' '.join(specs)}")
