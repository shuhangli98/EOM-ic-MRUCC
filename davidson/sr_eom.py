import itertools
import functools
import time
import numpy as np
import psi4
import forte
import forte.utils
import scipy.linalg
import scipy.optimize
from davidson import *
from nltgcr import *


def cc_residual_equations(op, ref, ham_op, exp_op, screen_thresh_H, screen_thresh_exp, maxk):
    """This function implements the CC residual equation

    Parameters
    ----------
    op : SparseOperator
        The cluster operator
    ref : StateVector
        The reference wave function
    ham_op : SparseHamiltonian
        The Hamiltonian operator
    exp_op : SparseExp
        The exponential operator        

    Returns
    -------
    tuple(list(float),float)
        A tuple with the residual and the energy
    """

    # Step 1. Compute exp(S)|Phi>
    wfn = exp_op.compute(op, ref, scaling_factor=1.0,
                         screen_thresh=screen_thresh_exp, maxk=maxk)

    # Step 2. Compute H exp(S)|Phi>
    Hwfn = ham_op.compute(wfn, screen_thresh_H)

    # Step 3. Compute exp(-S) H exp(S)|Phi>
    R = exp_op.compute(op, Hwfn, scaling_factor=-1.0,
                       screen_thresh=screen_thresh_exp, maxk=maxk)

    # Step 4. Project residual onto excited determinants: <Phi^{ab}_{ij}|R>
    residual = forte.get_projection(op, ref, R)

    # E = <ref|R>, R is a StateVector, which can be looked up by the determinant
    energy = 0.0
    for det, coeff in ref.items():
        energy += coeff * R[det]

    return (residual, energy)


def cc_variational_functional(t, op, ref, ham_op, exp_op, screen_thresh_H, screen_thresh_exp, maxk):
    '''
    E[A] = <Psi|exp(-A) H exp(A)|Psi>
    '''
    op.set_coefficients(t)
    # Step 1. Compute exp(S)|Phi>
    wfn = exp_op.compute(op, ref, scaling_factor=1.0,
                         screen_thresh=screen_thresh_exp, maxk=maxk)

    # Step 2. Compute H exp(S)|Phi>
    Hwfn = ham_op.compute(wfn, screen_thresh_H)

    # Step 3. Compute exp(-S) H exp(S)|Phi>
    R = exp_op.compute(op, Hwfn, scaling_factor=-1.0,
                       screen_thresh=screen_thresh_exp, maxk=maxk)

    # Step 4. Get the energy: <Phi|R>
    # E = <ref|R>, R is a StateVector, which can be looked up by the determinant
    energy = 0.0
    for det, coeff in ref.items():
        energy += coeff * R[det]

    norm = forte.overlap(wfn, wfn)
    return energy  # /norm


def update_amps(op, residual, denominators):
    """This function updates the CC amplitudes

    Parameters
    ----------
    op : SparseOperator
        The cluster operator. The amplitudes will be updates after running this function
    residual : list(float)
        The residual
    denominators : list(float)
        The Møller-Plesset denominators
    """
    t = op.coefficients()
    # update the amplitudes
    for i in range(op.size()):
        t[i] += residual[i] / denominators[i]
    # push new amplitudes to the T operator
    op.set_coefficients(t)


def sym_dir_prod(occ_list, sym_list):
    if (len(occ_list) == 0):
        return 0
    elif (len(occ_list) == 1):
        return sym_list[occ_list[0]]
    else:
        return functools.reduce(lambda i, j:  i ^ j, [sym_list[x] for x in occ_list])


SPIN_LABELS = {0: 'singlet', 3: 'doublet',
               8: 'triplet', 15: 'quartet', 24: 'quintet'}


class SR_EOM:
    def __init__(self, psi4_wfn, unitary, sym=0, verbose=True, screen_thresh_H=1e-9, screen_thresh_exp=1e-9, maxk=19):
        self.unitary = unitary
        self.verbose = verbose
        self.screen_thresh_H = screen_thresh_H
        self.screen_thresh_exp = screen_thresh_exp
        self.maxk = maxk

        # pass mo_spaces={} to treat all orbitals as active
        self.forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={'GAS1': list(
            psi4_wfn.doccpi().to_tuple()), 'GAS3': list((psi4_wfn.nmopi() - psi4_wfn.doccpi()).to_tuple())})
        self.ints = self.forte_objs['ints']
        self.as_ints = self.forte_objs['as_ints']
        self.scf_info = self.forte_objs['scf_info']
        self.mo_space_info = self.forte_objs['mo_space_info']

        # get the number of MOs and alpha/beta electrons per irrep
        self.nmo = self.mo_space_info.size('CORRELATED')
        self.nael = self.scf_info.doccpi().sum() + self.scf_info.soccpi().sum()
        self.nbel = self.scf_info.doccpi().sum()

        if self.nael != self.nbel:
            raise RuntimeError(
                'The number of alpha and beta electrons must be the same')

        print(f'Number of orbitals:        {self.nmo}')
        print(f'Number of alpha electrons: {self.nael}')
        print(f'Number of beta electrons:  {self.nbel}')

        # Symmetry stuff
        self.sym = sym  # target symmetry

        self.occ = self.mo_space_info.absolute_mo('GAS1')
        self.vir = self.mo_space_info.absolute_mo('GAS3')

        self.nmopi = psi4_wfn.nmopi().to_tuple()
        self.point_group = psi4_wfn.molecule().point_group().symbol()

        self.nirrep = self.mo_space_info.nirrep()

        self.naelpi = psi4_wfn.nalphapi().to_tuple()
        self.nbelpi = psi4_wfn.nbetapi().to_tuple()

        self.occ_sym = self.mo_space_info.symmetry('GAS1')
        self.vir_sym = self.mo_space_info.symmetry('GAS3')
        self.all_sym = self.mo_space_info.symmetry('CORRELATED')
        if (self.verbose):
            print(f'{self.occ_sym=}')
        if (self.verbose):
            print(f'{self.all_sym=}')
        if (self.verbose):
            print(f'{self.vir_sym=}')

        # Specify the occupation of the the Hartree–Fock determinant
        self.hfref = forte.Determinant()
        irrep_start = [sum(self.nmopi[:h]) for h in range(self.nirrep)]
        for h in range(self.nirrep):
            for i in range(self.naelpi[h]):
                self.hfref.set_alfa_bit(irrep_start[h] + i, True)
            for i in range(self.nbelpi[h]):
                self.hfref.set_beta_bit(irrep_start[h] + i, True)

        print(f'Reference determinant: {self.hfref.str(self.nmo)}')

    def make_T(self, max_exc):
        # Prepare the cluster operator (closed-shell case)
        # self.occ = list(range(self.nael))
        # self.vir = list(range(self.nael,self.nmo))

        print(f'Occupied orbitals: {self.occ}')
        print(f'Virtual orbitals:  {self.vir}')

        self.op = forte.SparseOperator(
            antihermitian=True) if self.unitary else forte.SparseOperator(antihermitian=False)

        self.ea = self.scf_info.epsilon_a()
        self.eb = self.scf_info.epsilon_a()

        self.denominators = []

        # loop over total excitation level
        for n in range(1, max_exc + 1):
            # loop over beta excitation level
            for nb in range(n + 1):
                na = n - nb
                # loop over alpha occupied
                for ao in itertools.combinations(self.occ, na):
                    ao_sym = sym_dir_prod(ao, self.all_sym)
                    # loop over alpha virtual
                    for av in itertools.combinations(self.vir, na):
                        av_sym = sym_dir_prod(av, self.all_sym)
                        # loop over beta occupied
                        for bo in itertools.combinations(self.occ, nb):
                            bo_sym = sym_dir_prod(bo, self.all_sym)
                            # loop over beta virtual
                            for bv in itertools.combinations(self.vir, nb):
                                bv_sym = sym_dir_prod(bv, self.all_sym)
                                if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym):
                                    # compute the denominators
                                    e_aocc = functools.reduce(
                                        lambda x, y: x + self.ea.get(y), ao, 0.0)
                                    e_avir = functools.reduce(
                                        lambda x, y: x + self.ea.get(y), av, 0.0)
                                    e_bocc = functools.reduce(
                                        lambda x, y: x + self.eb.get(y), bo, 0.0)
                                    e_bvir = functools.reduce(
                                        lambda x, y: x + self.eb.get(y), bv, 0.0)
                                    self.denominators.append(
                                        e_aocc + e_bocc - e_bvir - e_avir)

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
                                    # a_{ij..}^{ab..} * (t_{ij..}^{ab..} - t_{ab..}^{ij..})
                                    self.op.add_term(l, 0.0)

        print(f'==> Operator <==')
        print(f'Number of amplitudes: {self.op.size()}')
        print("\n".join(self.op.str()))

        self.ref = forte.StateVector({self.hfref: 1.0})
        self.ham_op = forte.SparseHamiltonian(self.as_ints)
        self.exp_op = forte.SparseExp()  # exp([])

    def run_ccn_variational(self):
        self.t = [0.0] * self.op.size()
        res = scipy.optimize.minimize(fun=cc_variational_functional, x0=self.t,
                                      args=(self.op, self.ref, self.ham_op, self.exp_op,
                                            self.screen_thresh_H, self.screen_thresh_exp, self.maxk),
                                      method='BFGS')
        print(res)

    def run_ccn(self, e_convergence=1e-8, max_cc_iter=100):
        start = time.time()

        # initialize T = 0
        self.t = [0.0] * self.op.size()
        self.op.set_coefficients(self.t)

        # initalize E = 0
        old_e = 0.0

        print('=================================================================')
        print('   Iteration     Energy (Eh)       Delta Energy (Eh)    Time (s)')
        print('-----------------------------------------------------------------')

        for iter in range(max_cc_iter):
            # 1. evaluate the CC residual equations
            residual, self.e_ccn = cc_residual_equations(
                self.op, self.ref, self.ham_op, self.exp_op, self.screen_thresh_H, self.screen_thresh_exp, self.maxk)

            # 2. update the CC equations
            update_amps(self.op, residual, self.denominators)

            # 3. print information
            print(
                f'{iter:9d} {self.e_ccn:20.12f} {self.e_ccn - old_e:20.12f} {time.time() - start:11.3f}')

            # 4. check for convergence of the energy
            if abs(self.e_ccn - old_e) < e_convergence:
                break
            old_e = self.e_ccn

        print('=================================================================')

        print(f' CCn energy (forte): {self.e_ccn:20.12f} [Eh]')

    def make_ee_eom_basis(self, max_exc):
        # Reference determinant (0 excitations)
        _ee_eom_basis = [forte.StateVector({self.hfref: 1.0})]

        for k in range(1, max_exc+1):  # k is the excitation level
            for ak in range(k+1):  # alpha excitation level
                bk = k - ak
                for ao in itertools.combinations(self.occ, self.nael-ak):
                    ao_sym = sym_dir_prod(ao, self.all_sym)
                    for av in itertools.combinations(self.vir, ak):
                        av_sym = sym_dir_prod(av, self.all_sym)
                        for bo in itertools.combinations(self.occ, self.nbel-bk):
                            bo_sym = sym_dir_prod(bo, self.all_sym)
                            for bv in itertools.combinations(self.vir, bk):
                                bv_sym = sym_dir_prod(bv, self.all_sym)
                                if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym):
                                    d = forte.Determinant()
                                    for i in ao:
                                        d.set_alfa_bit(i, True)
                                    for i in av:
                                        d.set_alfa_bit(i, True)
                                    for i in bo:
                                        d.set_beta_bit(i, True)
                                    for i in bv:
                                        d.set_beta_bit(i, True)
                                    _ee_eom_basis.append(
                                        forte.StateVector({d: 1.0}))

        print(f'Number of EE-EOM basis states: {len(_ee_eom_basis)}')

        return _ee_eom_basis

    def make_ip_eom_basis(self, max_exc):
        # IP-EOM-CCSD
        # R = [1, t_{i}^{}, t_{ij}^{a}]

        # Reference determinant (0 excitations)
        _ip_eom_basis = [forte.StateVector({self.hfref: 1.0})]

        for k in range(1, max_exc+1):  # k is the excitation level
            j = k - 1  # number of creation operators
            for ak in range(k+1):  # alpha excitation level
                bk = k - ak  # beta excitation level
                for aj in range(j+1):
                    bj = j - aj
                    for ao in itertools.combinations(self.occ, self.nael-ak):
                        ao_sym = sym_dir_prod(ao, self.all_sym)
                        for av in itertools.combinations(self.vir, aj):
                            av_sym = sym_dir_prod(av, self.all_sym)
                            for bo in itertools.combinations(self.occ, self.nbel-bk):
                                bo_sym = sym_dir_prod(bo, self.all_sym)
                                for bv in itertools.combinations(self.vir, bj):
                                    bv_sym = sym_dir_prod(bv, self.all_sym)
                                    if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym):
                                        d = forte.Determinant()
                                        for i in ao:
                                            d.set_alfa_bit(i, True)
                                        for i in av:
                                            d.set_alfa_bit(i, True)
                                        for i in bo:
                                            d.set_beta_bit(i, True)
                                        for i in bv:
                                            d.set_beta_bit(i, True)
                                        _ip_eom_basis.append(
                                            forte.StateVector({d: 1.0}))

        return _ip_eom_basis

    def make_ea_eom_basis(self, max_exc):
        # EA-EOM-CCSD
        # R = [1, t_{}^{a}, t_{i}^{ab}]

        # Reference determinant (0 excitations)
        _ea_eom_basis = [forte.StateVector({self.hfref: 1.0})]
        max_exc = 2

        for k in range(1, max_exc+1):  # k is the creation level (number of creation operators)
            j = k - 1  # number of annihilation operators
            for ak in range(k+1):  # alpha creation level
                bk = k - ak  # beta creation level
                for aj in range(j+1):  # alpha annihilation level
                    bj = j - aj  # beta annihilation level
                    for ao in itertools.combinations(self.occ, self.nael-aj):
                        ao_sym = sym_dir_prod(ao, self.all_sym)
                        for av in itertools.combinations(self.vir, ak):
                            av_sym = sym_dir_prod(av, self.all_sym)
                            for bo in itertools.combinations(self.occ, self.nbel-bj):
                                bo_sym = sym_dir_prod(bo, self.all_sym)
                                for bv in itertools.combinations(self.vir, bk):
                                    bv_sym = sym_dir_prod(bv, self.all_sym)
                                    if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym):
                                        d = forte.Determinant()
                                        for i in ao:
                                            d.set_alfa_bit(i, True)
                                        for i in av:
                                            d.set_alfa_bit(i, True)
                                        for i in bo:
                                            d.set_beta_bit(i, True)
                                        for i in bv:
                                            d.set_beta_bit(i, True)
                                        _ea_eom_basis.append(
                                            forte.StateVector({d: 1.0}))

        return _ea_eom_basis

    def make_hbar(self, dets, algo='oprod'):
        H = np.zeros((len(dets), len(dets)))
        if (algo == 'naive'):
            for i in range(len(dets)):
                for j in range(len(dets)):
                    # exp(S)|j>
                    wfn = self.exp_op.compute(
                        self.op, dets[j], scaling_factor=1.0, screen_thresh=self.screen_thresh_exp, maxk=self.maxk)
                    # H exp(S)|j>
                    Hwfn = self.ham_op.compute(wfn, self.screen_thresh_H)
                    # exp(-S) H exp(S)|j>
                    R = self.exp_op.compute(
                        self.op, Hwfn, scaling_factor=-1.0, screen_thresh=self.screen_thresh_exp, maxk=self.maxk)
                    # <i|exp(-S) H exp(S)|j>
                    H[i, j] = forte.overlap(dets[i], R)
        elif (algo == 'oprod'):
            _wfn_list = []
            _Hwfn_list = []

            for i in range(len(dets)):
                wfn = self.exp_op.compute(
                    self.op, dets[i], scaling_factor=1.0, maxk=self.maxk, screen_thresh=self.screen_thresh_exp)
                Hwfn = self.ham_op.compute(wfn, self.screen_thresh_H)
                _wfn_list.append(wfn)
                _Hwfn_list.append(Hwfn)

            for i in range(len(dets)):
                for j in range(len(dets)):
                    H[i, j] = forte.overlap(_wfn_list[i], _Hwfn_list[j])
                    H[j, i] = H[i, j]

        return H

    def run_eom(self, max_exc, mode, print_eigvals=True):
        if (mode == 'ip'):
            self.eom_basis = self.make_ip_eom_basis(max_exc)
        elif (mode == 'ea'):
            self.eom_basis = self.make_ea_eom_basis(max_exc)
        elif (mode == 'ee'):
            self.eom_basis = self.make_ee_eom_basis(max_exc)

        self.s2 = np.zeros((len(self.eom_basis),)*2)
        for i, ibasis in enumerate(self.eom_basis):
            for j, jbasis in enumerate(self.eom_basis):
                self.s2[i, j] = forte.spin2(
                    next(ibasis.items())[0], next(jbasis.items())[0])

        self.eom_hbar = self.make_hbar(self.eom_basis)

        if (self.unitary):
            davidson = Davidson(self.eom_hbar)
            self.eom_eigval, self.eom_eigvec = davidson.kernel(n_lowest=3)
            # self.eom_eigval, self.eom_eigvec = scipy.linalg.eigh(self.eom_hbar)
            self.eom_eigval -= self.eom_eigval[0]
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            print(f"{'#':^4} {'E_exc / Eh':^25} {'<S^2>':^10}  {'S':^5}")
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            for i in range(1, len(self.eom_eigval)):
                s2_val = self.eom_eigvec[:,
                                         i].T @ self.s2 @ self.eom_eigvec[:, i]
                s = round(2*(-1+np.sqrt(1+4*s2_val)))
                s /= 4
                print(
                    f'{i:^4d} {self.eom_eigval[i]:^25.12f} {abs(s2_val):^10.3f} {abs(s):^5.1f}')
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
        else:
            self.eom_eigval, self.eom_eigvec = scipy.linalg.eig(self.eom_hbar)
            self.eom_eigval_argsort = np.argsort(np.real(self.eom_eigval))
            print(self.eom_eigval[self.eom_eigval_argsort[0]])
            self.eom_eigval -= self.eom_eigval[self.eom_eigval_argsort[0]]
            self.eom_eigval = np.real(self.eom_eigval)
            if (print_eigvals):
                print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
                print(f"{'#':^4} {'E_exc / Eh':^25} {'<S^2>':^10}  {'S':^5}")
                print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
                for i in range(1, len(self.eom_eigval)):
                    s2_val = self.eom_eigvec[:, self.eom_eigval_argsort[i]
                                             ].T @ self.s2 @ self.eom_eigvec[:, self.eom_eigval_argsort[i]]
                    s = round(2*(-1+np.sqrt(1+4*s2_val)))
                    s /= 4
                    print(
                        f'{i:^4d} {self.eom_eigval[self.eom_eigval_argsort[i]]:^25.12f} {abs(s2_val):^10.3f} {abs(s):^5.1f}')
                print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")


if __name__ == '__main__':
    test = 1

    if (test == 1):
        # setup xyz geometry for BeH2
        geometry = """
        Be 0.0     0.0     0.0
        H  0   1.310011  0.0
        H  0   -1.310011  0.0
        symmetry c1
        """

        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='sto-6g', reference='rhf', options={'E_CONVERGENCE': 1.e-12})

        if psi4_wfn.nirrep() != 1:
            raise RuntimeError(
                'This code can only run in C1 symmetry. Add "symmetry C1" in your geometry section.')

        # eccsd = psi4.energy('ccsd/sto-6g')

        print(f'SCF Energy:  {escf:16.12f}')
        # print(f'CCSD Energy: {eccsd:16.12f}')

        ccsd_unitary = SR_EOM(psi4_wfn, unitary=True)
        ccsd_unitary.make_T(max_exc=2)
        ccsd_unitary.run_ccn()
        ccsd_unitary.run_eom(max_exc=2, mode='ee')
        # ee_eom_basis = ccsdt_unitary.make_ee_eom_basis(2)
        # hbar_ee_ccsdt_unitary = ccsdt_unitary.make_hbar(ee_eom_basis, algo='oprod')
        # res = scipy.linalg.eig(hbar_ee_ccsdt_unitary)
        # print(np.sort(np.real(res[0])-ccsdt_unitary.e_ccn)[1:10])
    elif (test == 2):
        geometry = """
        H 0.0 0.0 0.0
        H 0.0 0.0 1.0
        H 0.0 0.0 2.0
        H 0.0 0.0 3.0
        H 0.0 0.0 4.0
        H 0.0 0.0 5.1
        symmetry c1
        """
        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='sto-3g', reference='rhf', options={'E_CONVERGENCE': 1.e-12})
        if psi4_wfn.nirrep() != 1:
            raise RuntimeError(
                'This code can only run in C1 symmetry. Add "symmetry C1" in your geometry section.')

        eccsd = psi4.energy('ccsd/sto-3g')

        print(f'SCF Energy:  {escf:16.12f}')
        print(f'CCSD Energy: {eccsd:16.12f}')
        ccsd = SR_EOM(psi4_wfn, unitary=False)
        ccsd.make_T(max_exc=2)
        ccsd.run_ccn()
        ccsd.make_ee_eom_basis(2)
        hbar_ee_ccsd = ccsd.make_hbar(ccsd.ee_eom_basis)
        res = scipy.linalg.eig(hbar_ee_ccsd)
        print(np.sort(np.real(res[0])-ccsd.e_ccn)[1:10])
    elif (test == 3):
        geometry = """
        Be 0.0     0.0     0.0
        H  0   1.310011  0.0
        H  0   -1.310011  0.0
        symmetry d2h
        """

        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='sto-6g', reference='rhf', options={'E_CONVERGENCE': 1.e-12})
        eccsd = psi4.energy('ccsd/sto-6g')
        print(f'SCF Energy:  {escf:16.12f}')
        print(f'CCSD Energy: {eccsd:16.12f}')

        a = SR_EOM(psi4_wfn, unitary=False)
        a.make_T(max_exc=2)
        a.run_ccn()
        a.run_eom(2, mode='ee')
    elif (test == 4):
        # setup xyz geometry for linear H4
        geometry = """
        Be 0.0     0.0     0.0
        H  0   1.310011  0.0
        H  0   -1.310011  0.0
        symmetry c1
        """

        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='sto-6g', reference='rhf', options={'E_CONVERGENCE': 1.e-12})

        print(f'SCF Energy:  {escf:16.12f}')

        ccsd_unitary = SR_EOM(
            psi4_wfn, unitary=True, screen_thresh_H=1e-8, screen_thresh_exp=1e-8, maxk=8)
        ccsd_unitary.make_T(max_exc=3)
        ccsd_unitary.run_ccn()
        ccsd_unitary.make_ee_eom_basis(3)
        hbar_ee_ccsd_unitary = ccsd_unitary.make_hbar(
            ccsd_unitary.ee_eom_basis, algo='oprod')
        res = scipy.linalg.eig(hbar_ee_ccsd_unitary)
        print(np.sort(np.real(res[0])-ccsd_unitary.e_ccn)[1:10])
    elif (test == 5):
        geometry = """
        Be 0.0     0.0     0.0
        H  0   1.310011  0.0
        H  0   -1.310011  0.0
        symmetry d2h
        """

        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='sto-6g', reference='rhf', options={'E_CONVERGENCE': 1.e-12})
        eccsd = psi4.energy('ccsd/sto-6g')
        print(f'SCF Energy:  {escf:16.12f}')
        print(f'CCSD Energy: {eccsd:16.12f}')

        a = SR_EOM(psi4_wfn, unitary=True)
        a.make_T(max_exc=2)
        a.run_ccn_variational()
        a.run_ccn()
    elif (test == 6):
        geometry = """
        O
        H 1 1.0
        H 1 1.0 2 104.5
        symmetry c2v
        """
        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='cc-pvdz', reference='rhf', options={'E_CONVERGENCE': 1.e-12})
        eccsd = psi4.energy('ccsd/cc-pvdz')
        print(f'SCF Energy:  {escf:16.12f}')
        print(f'CCSD Energy: {eccsd:16.12f}')

        a = SR_EOM(psi4_wfn, unitary=False)
        a.make_T(max_exc=2)
        a.run_ccn()
        # a.make_ee_eom_basis(2)
        # hbar_ee_a = a.make_hbar(a.ee_eom_basis)
        # res = scipy.linalg.eig(hbar_ee_a)
        # print(np.sort(np.real(res[0])-a.e_ccn)[1:10])
    elif (test == 7):
        geometry = '''
        Ne
        symmetry d2h
        '''
        escf, psi4_wfn = forte.utils.psi4_scf(
            geometry, basis='cc-pvdz', reference='rhf', options={'E_CONVERGENCE': 1.e-12})
        eccsd = psi4.energy('ccsd/cc-pvdz')
        print(f'SCF Energy:  {escf:16.12f}')
        print(f'CCSD Energy: {eccsd:16.12f}')
        a = SR_EOM(psi4_wfn, unitary=False)
        a.make_T(max_exc=2)
        a.run_ccn()
        # a.make_ee_eom_basis(2)
        # hbar_ee_a = a.make_hbar(a.ee_eom_basis)
        # res = scipy.linalg.eig(hbar_ee_a)
        # print(np.sort(np.real(res[0])-a.e_ccn)[1:10])
