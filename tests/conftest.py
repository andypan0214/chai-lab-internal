"""Pytest path setup.

Makes chai_hpc/{input_builder,restraint_builder}.py importable (chai_hpc is
not a package, by design -- it mirrors the flat script layout in section 12
of the task), and, if a local clone of the official chai-lab repository is
present next to this project (see Step A: `_chai_src/`), makes the REAL
chai_lab.data.parsing.restraints module importable too, so a subset of the
restraint tests can round-trip our generated CSV through Chai's own parser
instead of only our re-implementation of its rules.
"""

import sys
from pathlib import Path

CHAI_HPC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAI_HPC_DIR.parent
CHAI_SRC_DIR = PROJECT_ROOT / "_chai_src"

if str(CHAI_HPC_DIR) not in sys.path:
    sys.path.insert(0, str(CHAI_HPC_DIR))

# Only used by tests that explicitly try to import chai_lab; harmless to add
# even if _chai_src doesn't exist on this machine.
if CHAI_SRC_DIR.is_dir() and str(CHAI_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CHAI_SRC_DIR))
