"""Exposes the repo's ``config/`` as a Hydra search path, so every primary config can include the
untracked ``config/paths/local.yaml`` with ``optional /paths: local``.

``hydra.searchpath`` in a YAML file is honoured only on the primary config, which would force the same
block into every config that inherits another."""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class LewmSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append("lewm-paths", f"file://{Path(__file__).resolve().parents[1] / 'config'}")
