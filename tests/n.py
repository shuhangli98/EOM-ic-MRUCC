import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_umrcc import *

psi4.core.set_output_file("n.dat", False)
psi4.set_num_threads(10)
mol = psi4.geometry(
    """
    0  4                
    N  0  0.0  0.0
    symmetry c1
    """
)
psi4.set_options(
    {
        "basis": "cc-pvdz",
        "frozen_docc": [0],
        "restricted_docc": [1],
        "reference": "rohf",
    }
)

forte_options = {
    "basis": "cc-pvdz",
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "frozen_docc": [0],
    "restricted_docc": [1],
    "active": [4],
    "root_sym": 0,
    "maxiter": 100,
    "e_convergence": 1e-8,
    "r_convergence": 1e-8,
    "casscf_e_convergence": 1e-8,
    "casscf_g_convergence": 1e-6,
}

E_casscf, wfn_cas = psi4.energy(
    "forte", forte_options=forte_options, return_wfn=True
)

print(f"CASSCF Energy = {E_casscf}")

mos_spaces = {"GAS1": [1], "GAS2": [4], "GAS3": [9]}

ic_mrcc = EOM_MRCC(
    mos_spaces,
    wfn_cas,
    verbose=True,
    maxk=8,
    screen_thresh_H=1e-8,
    screen_thresh_exp=1e-8,
)
ic_mrcc.get_casci_wfn([4, 1])
ic_mrcc.initialize_op()
ic_mrcc.run_ic_mrcc(
    e_convergence=1e-9, eta=-1.0, thres=1e-6, algo="oprod", max_cc_iter=100
)

ic_mrcc.run_eom_ee_mrcc([4, 1], thres=1e-6, algo="oprod", internal_max_exc=1)

psi4.core.clean()