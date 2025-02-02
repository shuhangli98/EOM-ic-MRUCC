import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_umrcc import *

psi4.core.set_output_file("h2o.dat", False)
geometry = """
    O  
    H  1   1.1  
    H  1   1.1  2 104
    symmetry c1
    """

escf, wfn_cas = forte.utils.psi4_scf(
    geometry,
    basis="sto-3g",
    reference="rhf",
    options={"E_CONVERGENCE": 1.0e-12},
)

mos_spaces = {"GAS1": [5], "GAS2": [0], "GAS3": [2]}

ic_mrcc = EOM_MRCC(
    mos_spaces,
    wfn_cas,
    verbose=False,
    unitary=True,
    maxk=8,
    screen_thresh_H=1e-8,
    screen_thresh_exp=1e-8,
)
ic_mrcc.get_casci_wfn([0, 0])
ic_mrcc.initialize_op()
ic_mrcc.run_ic_mrcc(
    e_convergence=1e-9, eta=-1.0, thres=1e-6, algo="oprod", max_cc_iter=0
)
ic_mrcc.run_eom_ee_mrcc([0, 0], thres=1e-6, algo="oprod")

