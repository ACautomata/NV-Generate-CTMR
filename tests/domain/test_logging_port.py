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

"""Contract tests for the logging port (ADR-0019 §3, #269).

The port names the logger face the application shells emit through (today:
``info``/``warning``). Any conforming sink is injectable -- the fake here is
the shape future application tests will use; the stdlib logger the
infrastructure ``setup_logging`` assembles satisfies the same port
structurally.
"""

import logging

from ctmr.domain.logging import Logger


class FakeLogger:
    """In-memory sink recording every port call."""

    def __init__(self):
        self.info_lines = []
        self.warning_lines = []

    def info(self, message):
        self.info_lines.append(message)

    def warning(self, message):
        self.warning_lines.append(message)


def test_the_fake_sink_satisfies_the_port():
    logger = FakeLogger()
    assert isinstance(logger, Logger)

    logger.info("hooked up")
    logger.warning("unexpected keys")
    assert logger.info_lines == ["hooked up"]
    assert logger.warning_lines == ["unexpected keys"]


def test_the_stdlib_logger_setup_logging_returns_satisfies_the_port():
    assert isinstance(logging.getLogger("anything"), Logger)

    from ctmr.infrastructure.maisi_engine.diff_model_setting import setup_logging

    assert isinstance(setup_logging("finetune"), Logger)
