import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_umrcc import *

# Symmetry test: edge case: 1 CAS electron per spin
psi4.core.set_output_file("beh2_22cas.dat", False)
x = 1.000
mol = psi4.geometry(
    f"""
    Be 0.0   0.0             0.0
    H  {x}   {2.54-0.46*x}   0.0
    H  {x}  -{2.54-0.46*x}   0.0
    symmetry c2v
    units bohr
    """
)

psi4.set_options(
    {
        "basis": "sto-6g",
        "frozen_docc": [0, 0, 0, 0],
        "restricted_docc": [2, 0, 0, 0],
        "reference": "rhf",
    }
)

forte_options = {
    "basis": "sto-6g",
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "frozen_docc": [0, 0, 0, 0],
    "restricted_docc": [2, 0, 0, 0],
    "active": [1, 0, 0, 1],
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

mos_spaces = {"GAS1": [2, 0, 0, 0], "GAS2": [1, 0, 0, 1], "GAS3": [1, 0, 1, 1]}

ic_mrcc = EOM_MRCC(
    mos_spaces,
    wfn_cas,
    verbose=True,
    maxk=8,
    screen_thresh_H=1e-12,
    screen_thresh_exp=1e-8,
    ortho="sokolov",
    add_int=False,
    cas_int=False,
    commutator=False,
    n_comm=2,
)
ic_mrcc.get_casci_wfn([1, 1])
ic_mrcc.initialize_op()
ic_mrcc.run_ic_mrcc(
    e_convergence=1e-9, max_cc_iter=200, eta=-1.0, thres=1e-4, algo="oprod"
)
# ic_mrcc.run_eom_ee_mrcc([1, 1], internal_max_exc=0, thres=1e-4, algo="oprod")
ic_mrcc.run_eom_ee_mrcc([1, 1], internal_max_exc=2, thres=1e-4, algo="oprod")
