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
assert np.isclose(ic_mrcc.e, -74.94207989868082, atol=1.0e-8)
ic_mrcc.run_eom_ee_mrcc([0, 0], thres=1e-6, algo="oprod")
assert np.isclose(ic_mrcc.eval_ic[1], -74.7058339380, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[2], -74.6574213703, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[3], -74.6187347560, atol=1.0e-8)
assert np.isclose(ic_mrcc.eval_ic[4], -74.6154946067, atol=1.0e-8)