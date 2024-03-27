from .solver import BaseSolver
from .det_solver import DetSolver
from .cl_solver import CLSolver

from typing import Dict

TASKS: Dict[str, BaseSolver] = {"detection": DetSolver, "cl": CLSolver}
