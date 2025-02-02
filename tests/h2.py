import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eom_umrcc import *


psi4.core.set_output_file("h2.dat", False)
geometry = """
    H  0  0.0  0.0
    H  0  0.0  1.0
    symmetry c1
    """

escf, wfn_cas = forte.utils.psi4_scf(
    geometry,
    basis="cc-pvdz",
    reference="rhf",
    options={"E_CONVERGENCE": 1.0e-12},
)

mos_spaces = {"GAS1": [0], "GAS2": [2], "GAS3": [8]}

ic_mrcc = EOM_MRCC(
    mos_spaces,
    wfn_cas,
    verbose=False,
    maxk=8,
    screen_thresh_H=1e-8,
    screen_thresh_exp=1e-8,
)
ic_mrcc.get_casci_wfn([1, 1])
ic_mrcc.initialize_op()
ic_mrcc.run_ic_mrcc(
    e_convergence=1e-9, eta=-1.0, thres=1e-6, algo="oprod", max_cc_iter=0
)
assert np.isclose(ic_mrcc.e, -1.1083377195359838, atol=1.0e-8)
ic_mrcc.run_eom_ee_mrcc([1, 1], thres=1e-6, algo="oprod")
assert np.isclose(ic_mrcc.eval_ic[1], -0.8768102810, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[2], -0.7028095568, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[3], -0.4882296866, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[4], -0.3636720423, atol=1.0e-8)
