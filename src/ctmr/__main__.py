"""``python -m ctmr`` execution form of the unified CLI (ADR-0015 §3).

The frozen-instrument argv pins ``<python> -m ctmr measure predict ...``
(ADR-0009 decision 3), so the package itself must be module-runnable; this
glue forwards to the same :class:`ctmr.cli.CtmrCli` the ``ctmr`` console
script serves.
"""

from ctmr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
