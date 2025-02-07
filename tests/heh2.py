import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_mrcc import *

psi4.core.set_output_file("heh2.dat", False)
mol = psi4.geometry(
    """
    H  0  0.0  0.0
    H  0  0.0  5.0
    He  2.0 0.0 0.0
    symmetry c1
    """
)
# He  1.0 0.0 0.0
psi4.set_options(
    {
        "basis": "6-31g",
        "frozen_docc": [0],
        "restricted_docc": [1],
        "reference": "rhf",
    }
)

forte_options = {
    "basis": "6-31g",
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "frozen_docc": [0],
    "restricted_docc": [1],
    "active": [2],
    "root_sym": 0,
    "maxiter": 100,
    "e_convergence": 1e-8,
    "r_convergence": 1e-8,
    "casscf_e_convergence": 1e-8,
    "casscf_g_convergence": 1e-6,
}

E_casscf, wfn_cas = psi4.energy("forte", forte_options=forte_options, return_wfn=True)

print(f"CASSCF Energy = {E_casscf}")

mos_spaces = {"GAS1": [1], "GAS2": [2], "GAS3": [3]}

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
ic_mrcc.run_ic_mrcc(e_convergence=1e-9, eta=-1.0, thres=1e-6, max_cc_iter=100)
assert np.isclose(ic_mrcc.e, -3.862656426787, atol=1.0e-8)
psi4.core.clean()
