import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_mrcc import *

psi4.core.set_output_file("beh2.dat", False)
geometry = """
    Be 0.0     0.0     0.0
    H  0   1.310011  0.0
    H  0   -1.310011  0.0
    symmetry c1
    """

escf, wfn_cas = forte.utils.psi4_scf(
    geometry,
    basis="sto-6g",
    reference="rhf",
    options={"E_CONVERGENCE": 1.0e-12},
)

mos_spaces = {"GAS1": [2], "GAS2": [1], "GAS3": [4]}

ic_mrcc = EOM_MRCC(
    mos_spaces,
    wfn_cas,
    verbose=True,
    maxk=8,
    screen_thresh_H=1e-8,
    screen_thresh_exp=1e-8,
)
ic_mrcc.get_casci_wfn([1, 1])
ic_mrcc.initialize_op()
ic_mrcc.run_ic_mrcc()
assert np.isclose(ic_mrcc.e, -15.759408790280, atol=1.0e-8)
