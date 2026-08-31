# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logger: the logging port the application emits through (ADR-0019 §3, #269).

The port names the face application shells and kernels actually call
(``info``/``warning``); any conforming sink is injectable. Protocol only --
the stdlib logger the infrastructure ``setup_logging`` assembles satisfies it
structurally, as does any in-memory test sink.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """The emit face of a run log: informational progress and non-fatal warnings."""

    def info(self, message: str) -> None:
        """Record an informational progress line."""
        ...

    def warning(self, message: str) -> None:
        """Record a non-fatal warning the run continues past."""
        ...
