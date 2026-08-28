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

"""L1 evaluate entry: coordinates frozen-run evidence into a controlled report.

Migrated verbatim from ``scripts/brats_l1_quantitative.py`` (#141); the
``selftest`` subcommand died in the migration (its assertions live as pytest
functions under tests/application/acceptance/quantitative). Reached as
``ctmr accept quantitative evaluate ...``.
"""

import argparse
import sys

from ctmr.application.acceptance.quantitative.evidence import (
    ControlledJsonReader,
    FeatureManifestReader,
    FrozenRunRecordReader,
    L1ReportWriter,
    NiftiVolumeReader,
    P3PairManifestReader,
)
from ctmr.application.acceptance.quantitative.fid import BootstrapProtocol, L1QuantitativeError
from ctmr.application.acceptance.quantitative.report import L1ReportProducer
from ctmr.domain.intensity_protocol import MRIntensityNormalizer


class L1EvaluationCommand:
    """Coordinates frozen-run evidence into a controlled L1 report."""

    def __init__(self, run_reader, feature_reader, pair_reader, writer):
        self._run_reader = run_reader
        self._feature_reader = feature_reader
        self._pair_reader = pair_reader
        self._writer = writer

    def evaluate(self, run_path, feature_path, pair_path, output_path, bootstrap_protocol):
        record = self._run_reader.read(run_path)
        features = self._feature_reader.read(feature_path)
        if record["phase"] == "P3" and pair_path is None:
            raise L1QuantitativeError("P3 L1 assessment requires --pairs with stage-0 and candidate same-case volumes")
        if record["phase"] != "P3" and pair_path is not None:
            raise L1QuantitativeError("--pairs applies only to P3 L1 assessment")
        observations = self._pair_reader.read(pair_path) if pair_path is not None else []
        report = L1ReportProducer(bootstrap_protocol).produce(
            record,
            self._run_reader.challenges(record),
            features.records,
            observations,
            features.protocol,
        )
        return self._writer.write(report, output_path)


def main(argv=None):
    """Run the quantitative evaluate verb (``ctmr accept quantitative evaluate``)."""
    parser = argparse.ArgumentParser(
        prog="ctmr accept quantitative evaluate",
        description="Write a candidate-bound quantitative acceptance report from controlled evidence.",
    )
    parser.add_argument("--run", required=True, help="frozen brats-phase-run record")
    parser.add_argument("--features", required=True, help="controlled brats-l1-features/1 manifest")
    parser.add_argument("--pairs", help="P3 only: stage-0/candidate/reference pair manifest")
    parser.add_argument("--output", required=True, help="controlled output report path")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    try:
        documents = ControlledJsonReader()
        report_path = L1EvaluationCommand(
            FrozenRunRecordReader(documents),
            FeatureManifestReader(documents),
            P3PairManifestReader(documents, NiftiVolumeReader(), MRIntensityNormalizer()),
            L1ReportWriter(),
        ).evaluate(
            args.run,
            args.features,
            args.pairs,
            args.output,
            BootstrapProtocol(resamples=args.bootstrap_resamples, seed=args.seed),
        )
        print(f"L1 report written -> {report_path}")
        return 0
    except L1QuantitativeError as error:
        print(f"L1 INPUT ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
