import itertools
import functools
import time
import numpy as np
import copy
import psi4
import forte, forte.utils
from forte import forte_options
import scipy, scipy.constants
import math

eh_to_ev = scipy.constants.physical_constants["Hartree energy in eV"][0]


def cc_residual_equations(op, ref, ham_op, exp_op, is_unitary, screen_thresh_H):
    """This function implements the CC residual equation

    Parameters
    ----------
    op : SparseOperator
        The cluster operator
    ref : SparseState
        The reference wave function
    ham_op : SparseOperator
        The Hamiltonian operator
    exp_op : SparseExp
        The exponential operator
    is_unitary : bool
        Whether the cluster operator is unitary

    Returns
    -------
    tuple(list(float),float)
        A tuple with the residual and the energy
    """

    # Step 1. Compute exp(S)|Phi>
    if is_unitary:
        wfn = exp_op.apply_antiherm(op, ref, scaling_factor=1.0)
    else:
        wfn = exp_op.apply_op(op, ref, scaling_factor=1.0)

    # Step 2. Compute H exp(S)|Phi>
    Hwfn = forte.apply_op(ham_op, wfn, screen_thresh_H)

    # Step 3. Compute exp(-S) H exp(S)|Phi>
    if is_unitary:
        R = exp_op.apply_antiherm(op, Hwfn, scaling_factor=-1.0)
    else:
        R = exp_op.apply_op(op, Hwfn, scaling_factor=-1.0)

    # Step 4. Project residual onto excited determinants: <Phi^{ab}_{ij}|R>
    residual = forte.get_projection(op, ref, R)
    energy = forte.overlap(ref, R).real

    return (residual, energy)


def cc_residual_equations_truncated(op, ref, ham_op, screen_thresh_H, n_comm):
    # This function is used to test the effect of truncation on the BCH expansion.
    operator = op.to_operator()  # op is SparseOperatorList, operator is SparseOperator
    Hwfn = forte.apply_op(ham_op, ref, screen_thresh_H)
    residual = forte.get_projection(op, ref, Hwfn)
    residual = np.array(residual)
    energy = forte.overlap(ref, Hwfn).real
    for k in range(1, n_comm + 1):
        for l in range(k + 1):
            wfn_comm = ref
            m = k - l
            for _ in range(m):
                wfn_comm = forte.apply_op(operator, wfn_comm)
            wfn_comm = forte.apply_op(ham_op, wfn_comm, screen_thresh_H)
            for _ in range(l):
                wfn_comm = forte.apply_op(operator, wfn_comm)

            factor = 1.0 if l % 2 == 0 else -1.0
            residual += (
                factor
                * np.array(forte.get_projection(op, ref, wfn_comm)).real
                / (math.factorial(l) * math.factorial(m))
            )
            energy += (
                factor
                * forte.overlap(ref, wfn_comm).real
                / (math.factorial(l) * math.factorial(m))
            )

    return (residual, energy)


def update_amps_orthogonal(
    residual,
    denominators,
    op,
    t,
    P,
    S,
    X,
    numnonred,
    update_radius=0.01,
    eta=0.1,
    diis=None,
):
    namps = S.shape[0]

    # 1. Form the residual in the orthonormal basis
    R1 = X.T @ residual

    # 2. Form the M in the nonredundant space
    # M = X^+ S
    M = X.T @ S
    # M = X^+ S Delta
    for i in range(namps):
        for j in range(namps):
            M[i][j] *= denominators[j] + eta
    # M = X^+ S Delta X
    M = M @ X

    # 3. Form the r' vector in the nonredundant space
    kn = -R1[:numnonred]
    # Form the M matrix in the nonredundant space
    Mn = M[:numnonred, :numnonred]

    # 4. Solve the linear equation
    dK_short = np.linalg.solve(Mn, kn)
    dK = np.zeros(namps)

    # 5. Update radius
    dK_short_norm = np.linalg.norm(dK_short)
    if dK_short_norm > update_radius:
        dK_normalized = update_radius * dK_short / dK_short_norm
        for x in range(numnonred):
            dK[x] = dK_normalized[x]
    else:
        for x in range(numnonred):
            dK[x] = dK_short[x]

    # 6. Calculate dT and set the new amplitudes.
    dT = X @ dK

    if diis is not None:
        t_old = copy.deepcopy(t)
        for x in range(len(t)):
            t[x] -= dT[x]
        t = diis.update(t, t_old)
    else:
        for x in range(len(t)):
            t[x] -= dT[x]

    t_proj = P @ t
    op.set_coefficients(list(t_proj))


def orthogonalization(ic_basis_full, thres=1e-6, distribution_print=False):
    # Evangelista and Gauss: full-orthogonalization.
    ic_basis = ic_basis_full[1:]
    namps = len(ic_basis)

    S = get_overlap(ic_basis)
    eigval, eigvec = np.linalg.eigh(S)

    if distribution_print:
        hist, bin_edges = np.histogram(eigval, bins=10 ** (np.arange(-14, 1.001, 1)))
        intervals = zip([0] + list(bin_edges[:-1]), bin_edges)

        for count, interval in zip(hist, intervals):
            print(f"Interval {interval}: {count} eigenvalues")

    numnonred = len(eigval[eigval > thres])
    X = np.zeros((namps, namps))
    U = eigvec[:, eigval > thres]
    S_diag_large = np.diag(1.0 / np.sqrt(eigval[eigval > thres]))
    X[:, :numnonred] = U @ S_diag_large
    P = U @ U.T

    return P, S, X, numnonred


def orthogonalization_sokolov(
    ic_basis_full, num_op, thres_single=1e-4, thres_double=1e-8
):
    # Double thresholds. No GNO.
    ic_basis_proj = ic_basis_full[: num_op[0] + num_op[1] + 1]
    num_op_proj = np.array([num_op[0], num_op[1]])
    P_proj, S_proj, X_proj, numnonred_proj = orthogonalization_projective(
        ic_basis_proj, num_op_proj, thres=thres_single
    )
    num_proj = P_proj.shape[0]

    ic_basis_direct = [ic_basis_full[0]] + ic_basis_full[num_op[0] + num_op[1] + 1 :]
    P_direct, S_direct, X_direct, numnonred_direct = orthogonalization(
        ic_basis_direct, thres=thres_double
    )
    num_direct = P_direct.shape[0]

    P = np.zeros((num_proj + num_direct, num_proj + num_direct))
    P[:num_proj, :num_proj] = P_proj
    P[num_proj:, num_proj:] = P_direct

    S = np.zeros((num_proj + num_direct, num_proj + num_direct))
    S[:num_proj, :num_proj] = S_proj
    S[num_proj:, num_proj:] = S_direct

    X = np.zeros((num_proj + num_direct, num_proj + num_direct))
    X[:num_proj, :numnonred_proj] = X_proj[:, :numnonred_proj]
    X[num_proj:, numnonred_proj : numnonred_proj + numnonred_direct] = X_direct[
        :, :numnonred_direct
    ]

    numnonred = numnonred_proj + numnonred_direct

    return P, S, X, numnonred


def orthogonalization_sokolov_direct(
    ic_basis_full, num_op, thres_single=1e-4, thres_double=1e-10
):
    # Double thresholds. No GNO.
    ic_basis_proj = ic_basis_full[: num_op[0] + num_op[1] + 1]
    P_proj, S_proj, X_proj, numnonred_proj = orthogonalization(
        ic_basis_proj, thres=thres_single
    )
    num_proj = P_proj.shape[0]

    ic_basis_direct = [ic_basis_full[0]] + ic_basis_full[num_op[0] + num_op[1] + 1 :]
    P_direct, S_direct, X_direct, numnonred_direct = orthogonalization(
        ic_basis_direct, thres=thres_double
    )
    num_direct = P_direct.shape[0]

    P = np.zeros((num_proj + num_direct, num_proj + num_direct))
    P[:num_proj, :num_proj] = P_proj
    P[num_proj:, num_proj:] = P_direct

    S = np.zeros((num_proj + num_direct, num_proj + num_direct))
    S[:num_proj, :num_proj] = S_proj
    S[num_proj:, num_proj:] = S_direct

    X = np.zeros((num_proj + num_direct, num_proj + num_direct))
    X[:num_proj, :numnonred_proj] = X_proj[:, :numnonred_proj]
    X[num_proj:, numnonred_proj : numnonred_proj + numnonred_direct] = X_direct[
        :, :numnonred_direct
    ]

    numnonred = numnonred_proj + numnonred_direct

    return P, S, X, numnonred


def orthogonalization_projective(ic_basis_full, num_op, thres=1e-6):
    # The ic_basis_full must bave block structure.
    # Sequential orthogonalization.
    ic_basis_1 = ic_basis_full[1 : num_op[0] + 1]
    ic_basis_2 = ic_basis_full[num_op[0] + 1 :]
    namps_1 = len(ic_basis_1)
    namps_2 = len(ic_basis_2)
    namps = namps_1 + namps_2
    S = get_overlap(ic_basis_full[1:])
    X = np.zeros((namps, namps))

    # 1. Orthogonalize the single excitation block.
    S1 = S[:namps_1, :namps_1].copy()
    eigval1, eigvec1 = np.linalg.eigh(S1)
    numnonred1 = len(eigval1[eigval1 > thres])
    X1 = np.zeros((namps, namps_1))
    U1_short = eigvec1[:, eigval1 > thres]
    U1 = np.zeros((namps, numnonred1))
    U1[:namps_1, :numnonred1] = U1_short.copy()
    S1_diag_large = np.diag(1.0 / np.sqrt(eigval1[eigval1 > thres]))
    X1[:namps_1, :numnonred1] = U1_short @ S1_diag_large

    # 2. Construct the Q matrix.
    Q = np.identity(namps)
    Q -= X1 @ X1.T @ S

    # 3. Construct the new S matrix with single excitation operatoras projected out.
    S2 = Q.T @ S @ Q

    # 4. Construct transformation matrix for the double excitation block.
    eigval2, eigvec2 = np.linalg.eigh(S2)
    U2 = eigvec2[:, eigval2 > thres]
    S2_diag_large = np.diag(1.0 / np.sqrt(eigval2[eigval2 > thres]))
    U2 = Q @ U2
    X2 = U2 @ S2_diag_large

    # 5. Concatenate the two transformation matrices.
    X12 = np.concatenate((X1, X2), axis=1)
    U = np.concatenate((U1, U2), axis=1)
    numnonred = X12.shape[1]
    X[:, :numnonred] = X12.copy()

    # 6. Construct the projection matrix.
    P = U @ U.T

    return P, S, X, numnonred


def orthogonalization_GNO(ic_basis_full, Y, thres=1e-6):
    ic_basis = ic_basis_full[1:]
    namps = len(ic_basis)
    # 1. Construct metric matrix.
    S = get_overlap(ic_basis)

    # 2. Tranform the metric matrix in the basis of GNO excitation operators.
    S_GNO = Y.T @ S @ Y
    eigval, eigvec = np.linalg.eigh(S_GNO)

    # 3. Construct transformed P and X matrices.
    numnonred = len(eigval[eigval > thres])
    X = np.zeros((namps, namps))
    U = eigvec[:, eigval > thres]
    # U = Y @ U # Transformed U matrix

    S_diag_large = np.diag(1.0 / np.sqrt(eigval[eigval > thres]))
    X[:, :numnonred] = Y @ U @ S_diag_large  # Transformed X matrix
    Y_inv = np.linalg.inv(Y)
    P = Y @ U @ U.T @ Y_inv  # Transformed P matrix
    return P, S, X, numnonred


def sym_dir_prod(occ_list, sym_list):
    # This function is used to calculate the symmetry of a specific excitation operator.
    if len(occ_list) == 0:
        return 0
    elif len(occ_list) == 1:
        return sym_list[occ_list[0]]
    else:
        return functools.reduce(lambda i, j: i ^ j, [sym_list[x] for x in occ_list])


def find_n_largest(arr, n):
    # Pair each element with its index
    indexed_arr = list(enumerate(arr))

    # Sort the array based on the elements (second item in each tuple)
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)

    # Extract the n largest elements and their indices
    n_largest_elements_and_indices = sorted_indexed_arr[:n]

    # Separate the elements and their indices into two lists
    elements = [elem for index, elem in n_largest_elements_and_indices]
    indices = [index for index, elem in n_largest_elements_and_indices]

    return list(zip(elements, indices))


def get_overlap(ic_basis):
    n = len(ic_basis)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = (forte.overlap(ic_basis[i], ic_basis[j])).real
            S[j, i] = S[i, j]
    return S


def get_spin2(ic_basis):
    s2 = np.zeros((len(ic_basis),) * 2)
    for i in range(len(ic_basis)):
        for j in range(i, len(ic_basis)):
            s2[i, j] = (forte.spin2(ic_basis[i], ic_basis[j])).real
            s2[j, i] = s2[i, j]
    return s2


class EOM_MRCC:
    def __init__(
        self,
        mos_spaces,
        wfn_cas,
        sym=0,
        max_exc=2,
        unitary=True,
        verbose=False,
        maxk=19,
        screen_thresh_H=0.0,
        screen_thresh_exp=1e-12,
        ortho="direct",
        cas_int=False,
        commutator=False,
        n_comm=2,
    ):
        self.forte_objs = forte.utils.prepare_forte_objects(wfn_cas, mos_spaces)
        self.wfn_cas = wfn_cas
        self.mos_spaces = mos_spaces
        self.ints = self.forte_objs["ints"]
        self.as_ints = self.forte_objs["as_ints"]
        self.scf_info = self.forte_objs["scf_info"]
        self.mo_space_info = self.forte_objs["mo_space_info"]

        self.verbose = verbose
        self.maxk = maxk
        self.screen_thresh_H = screen_thresh_H
        self.screen_thresh_exp = screen_thresh_exp

        # Define MO spaces.
        self.occ = self.mo_space_info.corr_absolute_mo("GAS1")
        self.act = self.mo_space_info.corr_absolute_mo("GAS2")
        self.vir = self.mo_space_info.corr_absolute_mo("GAS3")
        self.all_orb = self.mo_space_info.corr_absolute_mo("CORRELATED")

        if self.verbose:
            print(f"{self.occ=}")
        if self.verbose:
            print(f"{self.act=}")
        if self.verbose:
            print(f"{self.vir=}")
        if self.verbose:
            print(f"{self.all_orb=}")

        self.hole = self.occ + self.act
        self.particle = self.act + self.vir
        if self.verbose:
            print(f"{self.hole=}")
        if self.verbose:
            print(f"{self.particle=}")

        self.max_exc = max_exc

        self.unitary = unitary

        # Obtain symmetry information.
        self.sym = sym  # target symmetry
        self.act_sym = self.mo_space_info.symmetry("GAS2")
        self.vir_sym = self.mo_space_info.symmetry("GAS3")
        self.all_sym = self.mo_space_info.symmetry("CORRELATED")
        self.nirrep = self.mo_space_info.nirrep()
        self.nmopi = wfn_cas.nmopi()
        if self.verbose:
            print(f"{self.act_sym=}")
        if self.verbose:
            print(f"{self.all_sym=}")
        if self.verbose:
            print(f"{self.vir_sym=}")

        self.nael = wfn_cas.nalpha()
        self.nbel = wfn_cas.nbeta()

        # Obtain 1-RDM.
        _frozen_docc = (
            self.mos_spaces["FROZEN_DOCC"]
            if "FROZEN_DOCC" in self.mos_spaces
            else [0] * self.nirrep
        )
        mos_spaces_rdm = {
            "FROZEN_DOCC": _frozen_docc,
            "RESTRICTED_DOCC": mos_spaces["GAS1"],
            "ACTIVE": mos_spaces["GAS2"],
        }
        self.ints_rdms = forte.utils.prepare_ints_rdms(
            wfn_cas, mos_spaces_rdm, rdm_level=2
        )
        self.rdms = self.ints_rdms["rdms"]
        self.gamma1, self.gamma2 = forte.spinorbital_rdms(self.rdms)

        # Construct generalized Fock matrix, obtain orbital energies.
        self.get_fock_block()

        self.ea = []
        self.eb = []

        for i in range(self.f.shape[0]):
            if i % 2 == 0:
                self.ea.append(self.f[i, i])
            else:
                self.eb.append(self.f[i, i])

        if self.verbose:
            print(f"{self.ea=}")

        # Some additional parameters.
        self.ortho = ortho
        self.cas_int = cas_int
        self.commutator = commutator
        self.n_comm = n_comm

    def get_fock_block(self):
        self.f = forte.spinorbital_oei(self.ints, self.all_orb, self.all_orb)
        v = forte.spinorbital_tei(
            self.ints, self.all_orb, self.occ, self.all_orb, self.occ
        )
        self.f += np.einsum("piqi->pq", v)
        v = forte.spinorbital_tei(
            self.ints, self.all_orb, self.act, self.all_orb, self.act
        )
        self.f += np.einsum("piqj,ij->pq", v, self.gamma1)

    def get_casci_wfn(self, nelecas):
        # This function is used to obtain the CASCI wave function.
        self.dets = []
        corbs = self.mo_space_info.corr_absolute_mo("GAS1")
        aorbs = self.mo_space_info.corr_absolute_mo("GAS2")
        aorbs_rel = range(len(aorbs))

        if self.verbose:
            print(f"{corbs=}")
        if self.verbose:
            print(f"{aorbs=}")
        nact_ael = nelecas[0]
        nact_bel = nelecas[1]

        for astr in itertools.combinations(aorbs_rel, nact_ael):
            asym = sym_dir_prod(astr, self.act_sym)
            for bstr in itertools.combinations(aorbs_rel, nact_bel):
                bsym = sym_dir_prod(bstr, self.act_sym)
                if asym ^ bsym == self.sym:
                    d = forte.Determinant()
                    for i in corbs:
                        d.set_alfa_bit(i, True)
                    for i in corbs:
                        d.set_beta_bit(i, True)
                    for i in astr:
                        d.set_alfa_bit(aorbs[i], True)
                    for i in bstr:
                        d.set_beta_bit(aorbs[i], True)
                    self.dets.append(d)

        ndets = len(self.dets)
        print(f"Number of determinants: {ndets}")
        H = np.zeros((ndets, ndets))
        for i in range(len(self.dets)):
            for j in range(i + 1):
                H[i, j] = self.as_ints.slater_rules(self.dets[i], self.dets[j])

        # print(H)
        evals_casci, evecs_casci = np.linalg.eigh(H, "L")

        e_casci = (
            evals_casci[0]
            + self.as_ints.scalar_energy()
            + self.as_ints.nuclear_repulsion_energy()
        )
        self.e_casci = e_casci
        print(f"CASCI Energy = {e_casci}")
        print(
            evals_casci
            + self.as_ints.scalar_energy()
            + self.as_ints.nuclear_repulsion_energy()
        )

        c_casci_0 = evecs_casci[:, 0]

        # Get the reference CASCI state.
        self.psi = forte.SparseState(dict(zip(self.dets, c_casci_0)))

        _frozen_docc = (
            self.mos_spaces["FROZEN_DOCC"]
            if "FROZEN_DOCC" in self.mos_spaces
            else [0] * self.nirrep
        )
        mos_spaces_fci = {
            "FROZEN_DOCC": _frozen_docc,
            "GAS1": [0] * self.nirrep,
            "GAS2": list(np.array(self.nmopi.to_tuple()) - np.array(_frozen_docc)),
            "GAS3": [0] * self.nirrep,
        }

        forte_objs_fci = forte.utils.prepare_forte_objects(self.wfn_cas, mos_spaces_fci)
        as_ints_fci = forte_objs_fci["as_ints"]

        self.ham_op = forte.sparse_operator_hamiltonian(as_ints_fci)
        self.exp_op = forte.SparseExp(self.maxk, self.screen_thresh_exp)
        self.as_ints_fci = as_ints_fci

    def initialize_op(self):
        # Initialize excitation operators.
        self.op_A = forte.SparseOperatorList()  # For MRUCC
        self.op_T = forte.SparseOperatorList()  # For MRCC
        self.oprator_list = []
        self.denominators = []
        self.ic_basis = [self.psi]

        # The number of operators for each rank.
        self.num_op = np.zeros(self.max_exc, dtype=int)

        self.op_idx = []

        self.flip = []

        # loop over total excitation level
        for n in range(1, self.max_exc + 1):
            # loop over beta excitation level
            for nb in range(n + 1):
                na = n - nb
                # loop over alpha occupied
                for ao in itertools.combinations(self.hole, na):
                    ao_sym = sym_dir_prod(ao, self.all_sym)
                    # loop over alpha virtual
                    for av in itertools.combinations(self.particle, na):
                        av_sym = sym_dir_prod(av, self.all_sym)
                        # loop over beta occupied
                        for bo in itertools.combinations(self.hole, nb):
                            bo_sym = sym_dir_prod(bo, self.all_sym)
                            # loop over beta virtual
                            for bv in itertools.combinations(self.particle, nb):
                                bv_sym = sym_dir_prod(bv, self.all_sym)
                                if ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym:
                                    # create an operator from a list of tuples (creation, alpha, orb) where
                                    #   creation : bool (true = creation, false = annihilation)
                                    #   alpha    : bool (true = alpha, false = beta)
                                    #   orb      : int  (the index of the mo)
                                    l = []  # a list to hold the operator triplets
                                    for i in ao:
                                        # alpha occupied
                                        l.append((False, True, i))
                                    for i in bo:
                                        # beta occupied
                                        l.append((False, False, i))
                                    for a in reversed(bv):
                                        # beta virtual
                                        l.append((True, False, a))
                                    for a in reversed(av):
                                        # alpha virtual
                                        l.append((True, True, a))
                                    all_in_act = all(item[2] in self.act for item in l)
                                    if not all_in_act:
                                        A_op_temp = forte.SparseOperatorList()
                                        T_op_temp = forte.SparseOperatorList()
                                        # compute the denominators
                                        e_aocc = 0.0
                                        e_avir = 0.0
                                        e_bocc = 0.0
                                        e_bvir = 0.0
                                        for i in ao:
                                            e_aocc += self.ea[i]
                                        for i in av:
                                            e_avir += self.ea[i]
                                        for i in bo:
                                            e_bocc += self.eb[i]
                                        for i in bv:
                                            e_bvir += self.eb[i]
                                        # Reorder l to act, occ, vir, act. Only for double excitations.
                                        if self.ortho == "GNO":
                                            num_act = 0
                                            for item in l:
                                                if item[2] in self.act:
                                                    num_act += 1
                                            if n == 2 and num_act >= 2:
                                                num_act_o = 0
                                                num_act_v = 0
                                                pos_act_o = []
                                                pos_act_v = []
                                                for idx, item in enumerate(l[:2]):
                                                    if item[2] in self.act:
                                                        num_act_o += 1
                                                        pos_act_o.append(idx)
                                                for idx, item in enumerate(l[2:]):
                                                    if item[2] in self.act:
                                                        num_act_v += 1
                                                        pos_act_v.append(idx)
                                                if num_act_o == 1:
                                                    if pos_act_o[0] == 0:
                                                        continue
                                                    else:
                                                        l[:2] = [l[1], l[0]]
                                                if num_act_v == 1:
                                                    if pos_act_v[0] == 1:
                                                        continue
                                                    else:
                                                        l[2:] = [l[3], l[2]]

                                        idx = []
                                        for item in l:
                                            # (spin, orbital)
                                            idx.append((item[1], item[2]))
                                        self.op_idx.append(idx)

                                        denom = e_aocc + e_bocc - e_bvir - e_avir
                                        self.denominators.append(denom)
                                        A_op_temp.add_term(
                                            l,
                                            1.0,
                                            allow_reordering=True,
                                        )
                                        T_op_temp.add_term(
                                            l,
                                            1.0,
                                            allow_reordering=True,
                                        )

                                        if T_op_temp[0].real < 0.0:
                                            coeff = [1.0] * len(T_op_temp)
                                            T_op_temp.set_coefficients(coeff)
                                            self.flip.append(-1.0)
                                        else:
                                            self.flip.append(1.0)

                                        self.num_op[n - 1] += 1
                                        self.ic_basis.append(T_op_temp @ self.psi)
                                        self.oprator_list.append(
                                            T_op_temp.to_operator()
                                        )
                                        # a_{ij..}^{ab..} * (t_{ij..}^{ab..} - t_{ab..}^{ij..})
                                        self.op_A.add_term(
                                            l,
                                            1.0,
                                            allow_reordering=True,
                                        )
                                        self.op_T.add_term(
                                            l,
                                            1.0,
                                            allow_reordering=True,
                                        )

        if self.ortho == "GNO":
            _Y_full = np.zeros(
                (
                    len(self.hole) * 2,
                    len(self.particle) * 2,
                    len(self.particle) * 2,
                    len(self.particle) * 2,
                    len(self.hole) * 2,
                    len(self.hole) * 2,
                )
            )
            _I_occ = np.identity(len(self.occ) * 2)
            _I_act = np.identity(len(self.act) * 2)
            _I_vir = np.identity(len(self.vir) * 2)
            _ho = slice(0, len(self.occ) * 2)
            _ha = slice(len(self.occ) * 2, len(self.occ) * 2 + len(self.act) * 2)
            _pa = slice(0, len(self.act) * 2)
            _pv = slice(len(self.act) * 2, len(self.act) * 2 + len(self.vir) * 2)

            _hole_idx = {i: idx for idx, i in enumerate(self.hole)}
            _particle_idx = {i: idx for idx, i in enumerate(self.particle)}

            def _hole_spin_idx(s):
                return _hole_idx[s[1]] * 2 if s[0] else _hole_idx[s[1]] * 2 + 1

            def _particle_spin_idx(s):
                return _particle_idx[s[1]] * 2 if s[0] else _particle_idx[s[1]] * 2 + 1

            _Y_full[_ho, _pa, _pv, _pv, _ha, _ho] -= np.einsum(
                "ij,ba,vu->ivbauj", _I_occ, _I_vir, self.gamma1
            )
            _Y_full[_ha, _pa, _pv, _pv, _ha, _ha] -= np.einsum(
                "ba,uw,xv->uxbavw", _I_vir, _I_act, self.gamma1
            )
            _Y_full[_ha, _pa, _pv, _pv, _ha, _ha] += np.einsum(
                "ba,uv,xw->uxbavw", _I_vir, _I_act, self.gamma1
            )
            _Y_full[_ho, _pa, _pa, _pa, _ha, _ho] -= np.einsum(
                "ij,xu,wv->iwxuvj", _I_occ, _I_act, self.gamma1
            )
            _Y_full[_ho, _pa, _pa, _pa, _ha, _ho] += np.einsum(
                "ij,wu,xv->iwxuvj", _I_occ, _I_act, self.gamma1
            )

            GNO_Y_sub = np.zeros((self.num_op[0], self.num_op[1]))
            for isingle, s in enumerate(self.op_idx[: self.num_op[0]]):
                for idouble, d in enumerate(self.op_idx[self.num_op[0] :]):
                    GNO_Y_sub[isingle, idouble] = _Y_full[
                        _hole_spin_idx(s[0]),
                        _particle_spin_idx(d[3]),
                        _particle_spin_idx(d[2]),
                        _particle_spin_idx(s[1]),
                        _hole_spin_idx(d[0]),
                        _hole_spin_idx(d[1]),
                    ]

            for i in range(self.num_op[1]):
                GNO_Y_sub[:, i] *= self.flip[i + self.num_op[0]]

            self.GNO_Y = np.identity(self.num_op[0] + self.num_op[1])

            self.GNO_Y[: self.num_op[0], self.num_op[0] :] = GNO_Y_sub.copy()

        self.denominators = np.array(self.denominators)

        if self.verbose:
            print(f"Number of IC basis functions: {len(self.ic_basis)}")
        if self.verbose:
            print(f"Breakdown: {self.num_op}")

    def run_ic_mrcc(
        self,
        e_convergence=1.0e-12,
        max_cc_iter=500,
        eta=0.1,
        thres=1e-4,
        thres_double=1e-8,
    ):
        start = time.time()
        if self.unitary:
            op = self.op_A
        else:
            op = self.op_T

        # This is a full ic_basis which contains psi_casci.
        ic_basis = self.ic_basis

        # initialize T = 0
        self.t = [0.0] * len(op)
        op.set_coefficients(self.t)
        diis = DIIS(self.t, diis_start=3)
        # diis = None

        # initalize E = 0
        self.e = self.e_casci
        old_e = 0.0

        print("=================================================================")
        print("   Iteration     Energy (Eh)       Delta Energy (Eh)    Time (s)")
        print("-----------------------------------------------------------------")
        P, S, X, numnonred = self.orthogonalize_ic_mrcc(ic_basis, thres, thres_double)

        radius = 0.01

        for iter in range(max_cc_iter):
            # 1. evaluate the CC residual equations.
            if self.commutator:
                self.residual, self.e = cc_residual_equations_truncated(
                    op, self.psi, self.ham_op, self.screen_thresh_H, n_comm=self.n_comm
                )  # Truncated BCH expansion.
            else:
                self.residual, self.e = cc_residual_equations(
                    op,
                    self.psi,
                    self.ham_op,
                    self.exp_op,
                    self.unitary,
                    self.screen_thresh_H,
                )  # Full BCH expansion.
            if (self.e.real - old_e.real) > 0.0:
                if radius > 1e-7:
                    radius /= 2.0

            # 2. update the CC equations
            update_amps_orthogonal(
                self.residual,
                self.denominators,
                op,
                self.t,
                P,
                S,
                X,
                numnonred,
                update_radius=radius,
                eta=eta,
                diis=diis,
            )

            # 3. Form Heff
            Heff = self.form_ic_mrcc_heff(op)
            w, vr = scipy.linalg.eig(Heff)
            vr = np.real(vr)
            idx = np.argmin(np.real(w))
            self.psi = forte.SparseState(dict(zip(self.dets, vr[:, idx])))
            ic_basis_new = self.make_new_ic_basis()

            P, S, X, numnonred = self.orthogonalize_ic_mrcc(
                ic_basis_new, thres, thres_double
            )
            # 4. print information
            print(
                f"{iter:9d} {self.e:20.12f} {self.e - old_e:20.12f} {time.time() - start:11.3f}"
            )

            # 5. check for convergence of the energy
            self.ic_basis = ic_basis_new.copy()
            if abs(self.e - old_e) < e_convergence:
                self.psi_coeff = vr[:, idx]
                print(
                    "================================================================="
                )
                print(f" ic-MRCCSD energy: {self.e:20.12f} [Eh]")
                P, S, X, numnonred = orthogonalization(
                    ic_basis_new, thres=thres, distribution_print=False
                )
                if self.verbose:
                    print(f"Number of selected operators for ic-MRCCSD: {numnonred}")
                    print(f"Number of possible CAS internal: {len(w)-1}")

                if self.cas_int:
                    for i in range(len(w)):
                        if i != idx:
                            cas_psi = forte.SparseState(dict(zip(self.dets, vr[:, i])))
                            self.ic_basis.append(cas_psi)
                    if self.verbose:
                        print(
                            f" Number of ic_basis for EOM_UMRCC (CAS Internal): {len(self.ic_basis)}"
                        )

                break
            old_e = self.e

    def form_ic_mrcc_heff(self, op):
        Heff = np.zeros((len(self.dets), len(self.dets)))
        if self.commutator:
            operator = op.to_operator()
            _wfn_map_full = []
            _Hwfn_map_full = []
            for i in range(len(self.dets)):
                idet = forte.SparseState({self.dets[i]: 1.0})
                wfn_comm = idet
                Hwfn_comm = forte.apply_op(self.ham_op, wfn_comm, self.screen_thresh_H)
                _wfn_list = [wfn_comm]
                _Hwfn_list = [Hwfn_comm]
                for _ in range(self.n_comm):
                    wfn_comm = forte.apply_op(operator, wfn_comm)
                    Hwfn_comm = forte.apply_op(
                        self.ham_op, wfn_comm, self.screen_thresh_H
                    )
                    _wfn_list.append(wfn_comm)
                    _Hwfn_list.append(Hwfn_comm)

                _wfn_map_full.append(_wfn_list)
                _Hwfn_map_full.append(_Hwfn_list)

            for i in range(len(self.dets)):
                for j in range(i + 1):
                    energy = 0.0
                    for k in range(self.n_comm + 1):
                        for l in range(k + 1):
                            m = k - l
                            right_wfn = _Hwfn_map_full[i][m]
                            left_wfn = _wfn_map_full[j][l]
                            energy += forte.overlap(left_wfn, right_wfn).real / (
                                math.factorial(l) * math.factorial(m)
                            )
                    Heff[j, i] = energy
                    Heff[i, j] = energy
        else:
            if self.unitary:
                _wfn_list = []
                _Hwfn_list = []

                for i in range(len(self.dets)):
                    idet = forte.SparseState({self.dets[i]: 1.0})
                    wfn = self.exp_op.apply_antiherm(op, idet, scaling_factor=1.0)
                    Hwfn = forte.apply_op(self.ham_op, wfn, self.screen_thresh_H)
                    _wfn_list.append(wfn)
                    _Hwfn_list.append(Hwfn)

                for i in range(len(self.dets)):
                    for j in range(len(self.dets)):
                        Heff[i, j] = forte.overlap(_wfn_list[i], _Hwfn_list[j]).real
                        Heff[j, i] = Heff[i, j]
            else:
                Heff = np.zeros((len(self.dets), len(self.dets)))
                for i in range(len(self.dets)):
                    for j in range(len(self.dets)):
                        idet = forte.SparseState({self.dets[i]: 1.0})
                        jdet = forte.SparseState({self.dets[j]: 1.0})
                        wfn = self.exp_op.apply_op(op, jdet, scaling_factor=1.0)
                        Hwfn = forte.apply_op(self.ham_op, wfn, self.screen_thresh_H)
                        R = self.exp_op.apply_op(op, Hwfn, scaling_factor=-1.0)
                        Heff[i, j] = forte.overlap(idet, R)

        return Heff

    def orthogonalize_ic_mrcc(self, ic_basis_new, thres, thres_double):
        if self.ortho == "direct":
            P, S, X, numnonred = orthogonalization(ic_basis_new, thres=thres)
        elif self.ortho == "projective":
            P, S, X, numnonred = orthogonalization_projective(
                ic_basis_new, self.num_op, thres=thres
            )
        elif self.ortho == "GNO":
            P, S, X, numnonred = orthogonalization_GNO(
                ic_basis_new, self.GNO_Y, thres=thres
            )
        elif self.ortho == "sokolov":
            P, S, X, numnonred = orthogonalization_sokolov(
                ic_basis_new,
                self.num_op,
                thres_single=thres,
                thres_double=thres_double,
            )
        elif self.ortho == "sokolov_direct":
            P, S, X, numnonred = orthogonalization_sokolov_direct(
                ic_basis_new,
                self.num_op,
                thres_single=thres,
                thres_double=thres_double,
            )
        return P, S, X, numnonred

    def make_new_ic_basis(self):
        ic_basis_new = [self.psi]
        for i in range(len(self.oprator_list)):
            ic_basis_new.append(forte.apply_op(self.oprator_list[i], self.psi))
        return ic_basis_new

    def run_eom_ee_mrcc(
        self,
        nelecas,
        thres=1e-6,
        internal_max_exc=2,
        det_analysis=False,
    ):

        if not self.unitary:
            raise Exception("EOM is only available for unitary MRCC methods.")
        if internal_max_exc > 2:
            raise Exception("EOM is only available for single and double excitations.")

        self.make_eom_ic_basis(internal_max_exc, nelecas)

        if self.commutator:
            self.get_hbar_commutator()
        else:
            if self.unitary:
                self.get_hbar_oprod()

        S_full = get_overlap(self.ic_basis)

        if not self.cas_int:
            print("Now do transformation to GNO basis.")
            S_full = self.GNO_P.T @ S_full @ self.GNO_P
            self.Hbar_ic = self.GNO_P.T @ self.Hbar_ic @ self.GNO_P
        S_full = np.real(S_full)
        eigval, eigvec = np.linalg.eigh(S_full)

        numnonred = 0
        S = np.array([0])
        U = np.array([0])

        numnonred = len(eigval[eigval > thres])
        S = np.diag(1.0 / np.sqrt(eigval[eigval > thres]))
        U = eigvec[:, eigval > thres]

        print(f"Number of selected operators for EOM-UMRCCSD: {numnonred}")
        X_tilde = U @ S

        H_ic_tilde = X_tilde.T @ self.Hbar_ic @ X_tilde
        H_ic_tilde = np.real(H_ic_tilde)
        self.eval_ic, evec_ic = np.linalg.eigh(H_ic_tilde)

        s2 = get_spin2(self.ic_basis)

        c_total = X_tilde @ evec_ic

        norm = np.zeros((len(evec_ic)))
        for i in range(len(self.eval_ic)):
            norm[i] = c_total[:, i].T @ S_full @ c_total[:, i]

        n_sin = 0
        n_tri = 0
        n_quintet = 0
        print("=" * 90)
        print(f"{'EOM-ic-UMRCC summary':^90}")
        print("-" * 90)
        print(
            f"{'Root':<5} {'Energy (Eh)':<20} {'Exc energy (Eh)':<20} {'Exc energy (eV)':<20} {'Spin':<10} {'<S^2>':<10}"
        )
        print("-" * 90)

        for i in range(len(self.eval_ic)):
            ci = c_total[:, i]
            if det_analysis and i < 10:
                dets = []
                coeffs = []
                for p in range(len(ci)):
                    ic_basis = self.ic_basis[p]
                    for det, coeff in ic_basis.items():
                        if det in dets:
                            coeffs[dets.index(det)] += ci[p] * coeff
                        else:
                            dets.append(det)
                            coeffs.append(ci[p] * coeff)
                coeffs = np.array(coeffs)
                co_norm = np.linalg.norm(coeffs)
                coeffs /= co_norm
                n_largest = find_n_largest(abs(coeffs), 10)
                print(f"the norm {np.linalg.norm(coeffs)}")
                for ndet in range(5):
                    print(
                        f"det{dets[n_largest[ndet][1]], coeffs[n_largest[ndet][1]]} \n"
                    )

            spin2 = abs(ci.T @ s2 @ ci) / norm[i]

            if spin2 < 1.0:
                n_sin += 1
                spin = "singlet"
                serial = str(n_sin)
            elif 1.0 < spin2 < 3.0:
                n_tri += 1
                spin = "triplet"
                serial = str(n_tri)
            elif 4.0 < spin2 < 7.0:
                n_quintet += 1
                spin = "quintet"
                serial = str(n_quintet)
            else:  # Change this.
                spin = "unknown"
                serial = ""
            print(
                f"{i+1:<5} {self.eval_ic[i]:<20.10f} {(self.eval_ic[i]-self.e):<20.10f} {(self.eval_ic[i]-self.e)*eh_to_ev:<20.10f} {spin+' '+serial:<10} {spin2:<10.5f}"
            )
        print("=" * 90)

    def make_eom_ic_basis(self, internal_max_exc, nelecas):
        # Separate single and double ic_basis.
        self.ic_basis_single = [self.psi]
        self.ic_basis_double = []
        for x in range(self.num_op[0]):
            self.ic_basis_single.append(self.oprator_list[x] @ self.psi)
        for y in range(self.num_op[1]):
            self.ic_basis_double.append(
                self.oprator_list[y + self.num_op[0]] @ self.psi
            )

        diag_1 = list(np.zeros(len(self.ic_basis_single)))  # With Psi.
        diag_2 = list(np.zeros(len(self.ic_basis_double)))

        if not self.cas_int:
            print("Use internally internal excitations. No GNO yet.")

            self.op_idx_single = self.op_idx[: self.num_op[0]]
            self.op_idx_double = self.op_idx[self.num_op[0] :]
            self.flip_single = self.flip[: self.num_op[0]]
            self.flip_double = self.flip[self.num_op[0] :]

            for n in range(1, internal_max_exc + 1):
                # loop over beta excitation level
                max_nb = min(n, nelecas[1])
                for nb in range(max_nb + 1):
                    # We should at least have two electrons in active space.
                    na = n - nb
                    # loop over alpha occupied
                    for ao in itertools.combinations(self.act, na):
                        ao_sym = sym_dir_prod(ao, self.all_sym)
                        # loop over alpha virtual
                        for av in itertools.combinations(self.act, na):
                            av_sym = sym_dir_prod(av, self.all_sym)
                            # loop over beta occupied
                            for bo in itertools.combinations(self.act, nb):
                                bo_sym = sym_dir_prod(bo, self.all_sym)
                                # loop over beta virtual
                                for bv in itertools.combinations(self.act, nb):
                                    bv_sym = sym_dir_prod(bv, self.all_sym)
                                    if ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym:
                                        T_op_temp = forte.SparseOperatorList()
                                        l = []  # a list to hold the operator triplets
                                        for i in ao:
                                            # alpha occupied
                                            l.append((False, True, i))
                                        for i in bo:
                                            # beta occupied
                                            l.append((False, False, i))
                                        for a in reversed(bv):
                                            # beta virtual
                                            l.append((True, False, a))
                                        for a in reversed(av):
                                            # alpha virtual
                                            l.append((True, True, a))

                                        op = []
                                        for a in av:
                                            op.append(f"{a}a+")
                                        for a in bv:
                                            op.append(f"{a}b+")
                                        for i in reversed(bo):
                                            op.append(f"{i}b-")
                                        for i in reversed(ao):
                                            op.append(f"{i}a-")

                                        # No reordering in principle.
                                        T_op_temp.add_term(
                                            l, 1.0, allow_reordering=False
                                        )

                                        idx = []
                                        for item in l:
                                            # (spin, orbital)
                                            idx.append((item[1], item[2]))

                                        if n == 1:
                                            self.op_idx_single.append(idx)
                                            self.flip_single.append(1.0)
                                            self.ic_basis_single.append(
                                                T_op_temp @ self.psi
                                            )
                                            r1_idx_1 = 0
                                            r1_idx_2 = 0
                                            if l[0][1] == True:
                                                r1_idx_1 = 2 * self.act.index(l[0][2])
                                            else:
                                                r1_idx_1 = (
                                                    2 * self.act.index(l[0][2]) + 1
                                                )
                                            if l[1][1] == True:
                                                r1_idx_2 = 2 * self.act.index(l[1][2])
                                            else:
                                                r1_idx_2 = (
                                                    2 * self.act.index(l[1][2]) + 1
                                                )
                                            diag_1.append(
                                                -self.gamma1[r1_idx_1, r1_idx_2]
                                            )

                                        elif n == 2:
                                            self.op_idx_double.append(idx)
                                            self.flip_double.append(1.0)
                                            self.ic_basis_double.append(
                                                T_op_temp @ self.psi
                                            )
                                            r2_idx_1 = 0
                                            r2_idx_2 = 0
                                            r2_idx_3 = 0
                                            r2_idx_4 = 0
                                            if l[0][1] == True:
                                                r2_idx_1 = 2 * self.act.index(l[0][2])
                                            else:
                                                r2_idx_1 = (
                                                    2 * self.act.index(l[0][2]) + 1
                                                )
                                            if l[1][1] == True:
                                                r2_idx_2 = 2 * self.act.index(l[1][2])
                                            else:
                                                r2_idx_2 = (
                                                    2 * self.act.index(l[1][2]) + 1
                                                )
                                            if l[2][1] == True:
                                                r2_idx_3 = 2 * self.act.index(l[2][2])
                                            else:
                                                r2_idx_3 = (
                                                    2 * self.act.index(l[2][2]) + 1
                                                )
                                            if l[3][1] == True:
                                                r2_idx_4 = 2 * self.act.index(l[3][2])
                                            else:
                                                r2_idx_4 = (
                                                    2 * self.act.index(l[3][2]) + 1
                                                )
                                            diag_2.append(
                                                -self.gamma2[
                                                    r2_idx_4,
                                                    r2_idx_3,
                                                    r2_idx_1,
                                                    r2_idx_2,
                                                ]
                                                - 2
                                                * self.gamma1[r2_idx_1, r2_idx_3]
                                                * self.gamma1[r2_idx_2, r2_idx_4]
                                                + 2
                                                * self.gamma1[r2_idx_1, r2_idx_4]
                                                * self.gamma1[r2_idx_2, r2_idx_3]
                                            )

            n_single = len(self.ic_basis_single) - 1  # The first one is psi.
            n_double = len(self.ic_basis_double)

            n_single_int = n_single - self.num_op[0]
            n_double_int = n_double - self.num_op[1]

            self.ic_basis = self.ic_basis_single + self.ic_basis_double

            print(
                f"Number of used internal: {n_single_int+n_double_int, n_single_int, n_double_int}"
            )

            if self.verbose:
                print(
                    f"Number of EOM basis function without psi (breakdown): {n_single, n_double}"
                )

            self.op_idx = self.op_idx_single + self.op_idx_double
            self.flip = self.flip_single + self.flip_double

            # Generalized normal ordering.
            print("GNO starts.")
            self.gamma1_hp = np.zeros((len(self.hole) * 2, len(self.particle) * 2))
            self.gamma1_hp[len(self.occ) * 2 :, : len(self.act) * 2] = (
                self.gamma1.copy()
            )

            _P_full = np.zeros(
                (
                    len(self.hole) * 2,
                    len(self.particle) * 2,
                    len(self.particle) * 2,
                    len(self.particle) * 2,
                    len(self.hole) * 2,
                    len(self.hole) * 2,
                )
            )
            _I_hole = np.identity(len(self.hole) * 2)
            _I_particle = np.identity(len(self.particle) * 2)

            _hole_idx = {i: idx for idx, i in enumerate(self.hole)}
            _particle_idx = {i: idx for idx, i in enumerate(self.particle)}

            def _hole_spin_idx(s):
                return _hole_idx[s[1]] * 2 if s[0] else _hole_idx[s[1]] * 2 + 1

            def _particle_spin_idx(s):
                return _particle_idx[s[1]] * 2 if s[0] else _particle_idx[s[1]] * 2 + 1

            _P_full += np.einsum(
                "ik,ab,jc->ibcajk", _I_hole, _I_particle, self.gamma1_hp
            )
            _P_full -= np.einsum(
                "ik,ac,jb->ibcajk", _I_hole, _I_particle, self.gamma1_hp
            )
            _P_full += np.einsum(
                "ij,ac,kb->ibcajk", _I_hole, _I_particle, self.gamma1_hp
            )
            _P_full -= np.einsum(
                "ij,ab,kc->ibcajk", _I_hole, _I_particle, self.gamma1_hp
            )

            GNO_P_sub = np.zeros((n_single, n_double))
            for isingle, s in enumerate(self.op_idx[:n_single]):
                for idouble, d in enumerate(self.op_idx[n_single:]):
                    GNO_P_sub[isingle, idouble] = _P_full[
                        _hole_spin_idx(s[0]),
                        _particle_spin_idx(d[3]),
                        _particle_spin_idx(d[2]),
                        _particle_spin_idx(s[1]),
                        _hole_spin_idx(d[0]),
                        _hole_spin_idx(d[1]),
                    ]

            for i in range(n_double):
                GNO_P_sub[:, i] *= self.flip[i + n_single]

            diag_total = diag_1 + diag_2
            diag_total[0] = 1.0
            self.GNO_P = np.identity(len(self.ic_basis))
            self.GNO_P[1 : n_single + 1, 1 + n_single :] = GNO_P_sub.copy()
            self.GNO_P[0, :] = np.array(diag_total).copy()

            print("GNO ends.")
        print(f" Number of ic_basis for EOM_UMRCC: {len(self.ic_basis)}")

    def get_ic_coeff(self):
        self.ic_coeff = np.zeros((len(self.dets_fci), len(self.ic_basis)))
        self.dets_fci = np.array(self.dets_fci)

        for j in range(len(self.ic_basis)):
            for d, coeff in self.ic_basis[j].items():
                loc = np.where(self.dets_fci == d)
                self.ic_coeff[loc, j] = coeff

    def get_hbar_commutator(self):
        _wfn_map_full = []
        _Hwfn_map_full = []
        self.Hbar_ic = np.zeros((len(self.ic_basis),) * 2)
        for ibasis in range(len(self.ic_basis)):
            i = self.ic_basis[ibasis]
            wfn_comm = i
            Hwfn_comm = forte.apply_op(self.ham_op, wfn_comm, self.screen_thresh_H)
            _wfn_list = [wfn_comm]
            _Hwfn_list = [Hwfn_comm]
            for _ in range(self.n_comm):
                wfn_comm = forte.apply_op(self.op_A, wfn_comm)
                Hwfn_comm = forte.apply_op(self.ham_op, wfn_comm, self.screen_thresh_H)
                _wfn_list.append(wfn_comm)
                _Hwfn_list.append(Hwfn_comm)

            _wfn_map_full.append(_wfn_list)
            _Hwfn_map_full.append(_Hwfn_list)

        for ibasis in range(len(self.ic_basis)):
            for jbasis in range(ibasis + 1):
                energy = 0.0
                for k in range(self.n_comm + 1):
                    for l in range(k + 1):
                        m = k - l
                        right_wfn = _Hwfn_map_full[ibasis][m]
                        left_wfn = _wfn_map_full[jbasis][l]
                        energy += forte.overlap(left_wfn, right_wfn).real / (
                            math.factorial(l) * math.factorial(m)
                        )
                self.Hbar_ic[jbasis, ibasis] = energy
                self.Hbar_ic[ibasis, jbasis] = energy

    def get_hbar_oprod(self):
        self.Hbar_ic = np.zeros((len(self.ic_basis),) * 2)
        _wfn_list = []
        _Hwfn_list = []

        for ibasis in range(len(self.ic_basis)):
            i = self.ic_basis[ibasis]
            if self.unitary:
                wfn = self.exp_op.apply_antiherm(self.op_A, i, scaling_factor=1.0)
            else:
                wfn = self.exp_op.apply_op(self.op_A, i, scaling_factor=1.0)
            Hwfn = forte.apply_op(self.ham_op, wfn, self.screen_thresh_H)
            _wfn_list.append(wfn)
            _Hwfn_list.append(Hwfn)

        for i in range(len(self.ic_basis)):
            for j in range(len(self.ic_basis)):
                self.Hbar_ic[i, j] = forte.overlap(_wfn_list[i], _Hwfn_list[j]).real
                self.Hbar_ic[j, i] = self.Hbar_ic[i, j]


class DIIS:
    """A class that implements DIIS for CC theory

        Parameters
    ----------
    diis_start : int
        Start the iterations when the DIIS dimension is greather than this parameter (default = 3)
    """

    def __init__(self, t, diis_start=3):
        self.t_diis = [t]
        self.e_diis = []
        self.diis_start = diis_start

    def update(self, t, t_old):
        """Update the DIIS object and return extrapolted amplitudes

        Parameters
        ----------
        t : list
            The updated amplitudes
        t_old : list
            The previous set of amplitudes
        Returns
        -------
        list
            The extrapolated amplitudes
        """

        if self.diis_start == -1:
            return t

        self.t_diis.append(t)
        self.e_diis.append(np.subtract(t, t_old))

        diis_dim = len(self.t_diis) - 1
        if (diis_dim >= self.diis_start) and (diis_dim < len(t)):
            # consturct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim + 1, diis_dim + 1)) * -1.0
            bsol = np.zeros(diis_dim + 1)
            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i in range(len(self.e_diis)):
                for j in range(i, len(self.e_diis)):
                    B[i, j] = np.dot(np.real(self.e_diis[i]), np.real(self.e_diis[j]))
                    if i != j:
                        B[j, i] = B[i, j]
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            x = np.linalg.solve(B, bsol)
            t_new = np.zeros((len(t)))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(self.t_diis[l + 1])
                t_new = np.add(t_new, temp_ary)
            return copy.deepcopy(list(np.real(t_new)))

        return t
