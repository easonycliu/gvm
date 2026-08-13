"""Selection helpers for timestamped experiment results."""

import re
from collections.abc import Callable
from pathlib import Path


RESULT_FILE = re.compile(
    r"^(?P<application>[^-]+)-(?P<label>.+)-"
    r"(?P<timestamp>\d{8}-\d{6})\.(?:json|txt)$"
)

LEGACY_LABELS = {
    "GVMDYN": "GVMFT",
    "GVMCOOP": "GVMAC",
}


def canonical_label(label: str) -> str:
    """Translate labels emitted by older experiment scripts."""
    head, separator, tail = label.partition("-")
    head = LEGACY_LABELS.get(head, head)
    return head + (separator + tail if separator else "")


def latest_result(
    input_path: str, application: str, requested_label: str
) -> Path | None:
    """Return the newest result for one application and exact label.

    Standalone LC and BE runs intentionally have different timestamps, so
    they cannot use the complete-pair selection applied to colocated runs.
    """
    newest: tuple[str, Path] | None = None
    for path in Path(input_path).iterdir():
        if not path.is_file():
            continue
        match = RESULT_FILE.match(path.name)
        if match is None or match.group("application") != application:
            continue
        if canonical_label(match.group("label")) != requested_label:
            continue
        timestamp = match.group("timestamp")
        if newest is None or timestamp > newest[0]:
            newest = (timestamp, path)
    return newest[1] if newest else None


def latest_complete_results(
    input_path: str,
    applications: list[str],
    normalize_label: Callable[[str], str] = lambda label: label,
) -> dict[str, dict[str, Path]]:
    """Return the newest complete application set for every data point.

    Results belonging to one run share a label and timestamp. A run is only
    eligible when it contains a result for every application, preventing a
    plot from accidentally combining files from two different runs.
    ``normalize_label`` controls what constitutes a plotted data point.
    """
    expected = set(applications)
    runs: dict[tuple[str, str], dict[str, Path]] = {}

    for path in Path(input_path).iterdir():
        if not path.is_file():
            continue
        match = RESULT_FILE.match(path.name)
        if match is None:
            continue
        application = match.group("application")
        if application not in expected:
            continue
        label = canonical_label(match.group("label"))
        key = (label, match.group("timestamp"))
        runs.setdefault(key, {})[application] = path

    selected: dict[str, tuple[str, dict[str, Path]]] = {}
    for (label, timestamp), files in runs.items():
        if set(files) != expected:
            continue
        point = normalize_label(label)
        if point not in selected or timestamp > selected[point][0]:
            selected[point] = (timestamp, files)

    if not selected:
        raise ValueError(
            f"No complete timestamped result sets for {applications} "
            f"were found in {input_path}"
        )

    return {point: files for point, (_, files) in selected.items()}
