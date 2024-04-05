import itertools
import functools
import time
import numpy as np
import copy
import psi4
import forte
import forte.utils
from forte import forte_options
import scipy
from functools import wraps
import math

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
    wfn = exp_op.compute(op, ref, scaling_factor=1.0, screen_thresh=screen_thresh_exp, maxk=maxk)
    
    # Step 2. Compute H exp(S)|Phi>
    Hwfn = ham_op.compute(wfn, screen_thresh_H)
    
    # Step 3. Compute exp(-S) H exp(S)|Phi>
    R = exp_op.compute(op, Hwfn, scaling_factor=-1.0, screen_thresh=screen_thresh_exp, maxk=maxk)
    
    # Step 4. Project residual onto excited determinants: <Phi^{ab}_{ij}|R>
    residual = forte.get_projection(op, ref, R)
    
    # E = <ref|R>, R is a StateVector, which can be looked up by the determinant
    energy = 0.0
    for det,coeff in ref.items():
        energy += coeff * R[det]

    return (residual, energy)

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

SPIN_LABELS = {0:'singlet', 3:'doublet', 8:'triplet', 15:'quartet', 24:'quintet'} 


class EOM_CC:
    def __init__(self, mos_spaces, wfn_cas,active_docc, ref_sym=0, max_exc=2, unitary=False, verbose=False, screen_thresh_H=1e-9, screen_thresh_exp=1e-9, maxk=19):
        self.forte_objs = forte.utils.prepare_forte_objects(wfn_cas,mos_spaces)
        self.mos_spaces = mos_spaces
        self.ints = self.forte_objs['ints']
        self.as_ints = self.forte_objs['as_ints']
        self.scf_info = self.forte_objs['scf_info']
        self.mo_space_info = self.forte_objs['mo_space_info']

        self.wfn_cas = wfn_cas
        self.verbose = verbose
        self.maxk = maxk
        self.screen_thresh_H = screen_thresh_H
        self.screen_thresh_exp = screen_thresh_exp
        
        # Define MO spaces.
        self.occ_cas = self.mo_space_info.corr_absolute_mo('GAS1')
        self.act_cas = self.mo_space_info.corr_absolute_mo('GAS2')
        self.vir_cas = self.mo_space_info.corr_absolute_mo('GAS3')
        self.all_orb = self.mo_space_info.corr_absolute_mo('CORRELATED')
        self.active_docc = active_docc
        
        if (self.verbose): print(f'{self.occ_cas=}')
        if (self.verbose): print(f'{self.act_cas=}')
        if (self.verbose): print(f'{self.vir_cas=}')
        
        self.max_exc = max_exc
        
        self.unitary = unitary
        
        # Obtain symmetry information.
        self.refsym = ref_sym  # This should always be 0.
        self.act_cas_sym = self.mo_space_info.symmetry('GAS2')
        self.vir_cas_sym = self.mo_space_info.symmetry('GAS3')
        self.all_sym = self.mo_space_info.symmetry('CORRELATED') 
        self.nirrep = self.mo_space_info.nirrep()
        self.nmopi = wfn_cas.nmopi()
        if (self.verbose): print(f'{self.act_cas_sym=}')
        if (self.verbose): print(f'{self.all_sym=}')
        if (self.verbose): print(f'{self.vir_cas_sym=}')

        self.nael = wfn_cas.nalpha()
        self.nbel = wfn_cas.nbeta()
        
        self.ea = self.scf_info.epsilon_a()
        self.eb = self.scf_info.epsilon_a()
        
        self.occ = []
        self.vir = []
        d = forte.Determinant()
        for i in self.occ_cas: 
            d.set_alfa_bit(i, True)
            d.set_beta_bit(i, True)
            self.occ.append(i)
        for i in self.active_docc: 
            d.set_alfa_bit(self.act_cas[i], True)
            d.set_beta_bit(self.act_cas[i], True)
            self.occ.append(self.act_cas[i])
            
            
        for i in range(len(self.act_cas)):
            if (i not in self.active_docc):
                self.vir.append(self.act_cas[i])
                
        for i in self.vir_cas: 
            self.vir.append(i)
        
        if (self.verbose): print(f'{self.occ=}')
        if (self.verbose): print(f'{self.vir=}')
    
        self.ref = forte.StateVector({d :1.0})
        print(f'The reference determinant is {d.str(len(self.all_orb))}.')
        
        ref_energy = self.as_ints.slater_rules(d,d) + self.as_ints.scalar_energy() + self.as_ints.nuclear_repulsion_energy()
        
        print(f'The reference deteminant energy = {ref_energy}.')
    
    def make_T(self):
        self.op = forte.SparseOperator(antihermitian=True) if self.unitary else forte.SparseOperator(antihermitian=False)
        self.denominators = []
        
        # loop over total excitation level
        for n in range(1,self.max_exc + 1):
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
                                if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.refsym):
                                    # compute the denominators
                                    e_aocc = functools.reduce(lambda x, y: x + self.ea.get(y),ao,0.0)
                                    e_avir = functools.reduce(lambda x, y: x + self.ea.get(y),av,0.0)
                                    e_bocc = functools.reduce(lambda x, y: x + self.eb.get(y),bo,0.0)
                                    e_bvir = functools.reduce(lambda x, y: x + self.eb.get(y),bv,0.0)
                                    self.denominators.append(e_aocc + e_bocc - e_bvir - e_avir)
                                    
                                    # create an operator from a list of tuples (creation, alpha, orb) where
                                    #   creation : bool (true = creation, false = annihilation)
                                    #   alpha    : bool (true = alpha, false = beta)
                                    #   orb      : int  (the index of the mo)
                                    l = [] # a list to hold the operator triplets
                                    for i in ao: l.append((False,True,i)) # alpha occupied
                                    for i in bo: l.append((False,False,i)) # beta occupied        
                                    for a in reversed(bv): l.append((True,False,a)) # beta virtual                                                                    
                                    for a in reversed(av): l.append((True,True,a)) # alpha virtual
                                    self.op.add_term(l,0.0) # a_{ij..}^{ab..} * (t_{ij..}^{ab..} - t_{ab..}^{ij..})

        print(f'==> Operator <==')
        print(f'Number of amplitudes: {self.op.size()}')
        print("\n".join(self.op.str()))
        
        self.ham_op = forte.SparseHamiltonian(self.as_ints)
        self.exp_op = forte.SparseExp() # exp([])
        
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
            residual, self.e_ccn = cc_residual_equations(self.op,self.ref,self.ham_op,self.exp_op,self.screen_thresh_H,self.screen_thresh_exp,self.maxk)
            
            # 2. update the CC equations
            update_amps(self.op,residual,self.denominators)
                
            # 3. print information
            print(f'{iter:9d} {self.e_ccn:20.12f} {self.e_ccn - old_e:20.12f} {time.time() - start:11.3f}')          
                
            # 4. check for convergence of the energy
            if abs(self.e_ccn - old_e) < e_convergence:
                break
            old_e = self.e_ccn
            
        print('=================================================================')

        print(f' CCn energy (forte): {self.e_ccn:20.12f} [Eh]')
        
    def make_eom_basis(self, sym):
        _ee_eom_basis = [self.ref] # Reference determinant (0 excitations)

        for k in range(1, self.max_exc+1): # k is the excitation level
            for ak in range(k+1): # alpha excitation level
                bk = k - ak
                for ao in itertools.combinations(self.occ, self.nael-ak):
                    ao_sym = sym_dir_prod(ao, self.all_sym)
                    for av in itertools.combinations(self.vir, ak):
                        av_sym = sym_dir_prod(av, self.all_sym)
                        for bo in itertools.combinations(self.occ, self.nbel-bk):
                            bo_sym = sym_dir_prod(bo, self.all_sym)
                            for bv in itertools.combinations(self.vir, bk):
                                bv_sym = sym_dir_prod(bv, self.all_sym)
                                if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == sym):
                                    d = forte.Determinant()
                                    for i in ao: d.set_alfa_bit(i, True)
                                    for i in av: d.set_alfa_bit(i, True)
                                    for i in bo: d.set_beta_bit(i, True)
                                    for i in bv: d.set_beta_bit(i, True)
                                    _ee_eom_basis.append(forte.StateVector({d:1.0}))
        
        
        print(f'Number of EOM-CC basis states (The reference is included) for sym {sym}: {len(_ee_eom_basis)}') 
    
        return _ee_eom_basis

    def make_hbar(self, dets, algo='naive'):
        H = np.zeros((len(dets),len(dets)))
        if (algo == 'naive'):
            for i in range(len(dets)):
                for j in range(len(dets)):
                    # exp(S)|j>
                    wfn = self.exp_op.compute(self.op,dets[j],scaling_factor=1.0,screen_thresh=self.screen_thresh_exp,maxk=self.maxk)
                    # H exp(S)|j>
                    Hwfn = self.ham_op.compute(wfn,self.screen_thresh_H)
                    # exp(-S) H exp(S)|j>
                    R = self.exp_op.compute(self.op,Hwfn,scaling_factor=-1.0,screen_thresh=self.screen_thresh_exp,maxk=self.maxk)
                    # <i|exp(-S) H exp(S)|j>
                    H[i,j] = forte.overlap(dets[i],R)
        elif (algo == 'oprod'): # This can only be used for the unitary coupled cluster.
            _wfn_list = []
            _Hwfn_list = []

            for i in range(len(dets)):
                wfn = self.exp_op.compute(self.op,dets[i],scaling_factor=1.0,maxk=self.maxk,screen_thresh=self.screen_thresh_exp)
                Hwfn = self.ham_op.compute(wfn,self.screen_thresh_H)
                _wfn_list.append(wfn)
                _Hwfn_list.append(Hwfn)

            for i in range(len(dets)):
                for j in range(len(dets)):
                    H[i,j] = forte.overlap(_wfn_list[i],_Hwfn_list[j])
                    H[j,i] = H[i,j]

        return H
        
    def run_eom(self, sym = 0, print_eigvals=True):
        self.eom_basis = self.make_eom_basis(sym = sym)
        self.s2 = np.zeros((len(self.eom_basis),)*2)
        for i,ibasis in enumerate(self.eom_basis):
            for j,jbasis in enumerate(self.eom_basis):
                self.s2[i,j] = forte.spin2(next(ibasis.items())[0],next(jbasis.items())[0])

        self.eom_hbar = self.make_hbar(self.eom_basis, algo='naive')
        self.eom_eigval, self.eom_eigvec = scipy.linalg.eig(self.eom_hbar)
        self.eom_eigval_argsort = np.argsort(np.real(self.eom_eigval))
        print(np.real(self.eom_eigval[self.eom_eigval_argsort[0]]))
        # self.eom_eigval -= self.eom_eigval[self.eom_eigval_argsort[0]]
        self.eom_eigval = np.real(self.eom_eigval)
        if (print_eigvals):
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            print(f"{'#':^4} {'E_exc / Eh':^25} {'<S^2>':^10}  {'S':^5}")
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            for i in range(1,len(self.eom_eigval)):
                s2_val = self.eom_eigvec[:,self.eom_eigval_argsort[i]].T @ self.s2 @ self.eom_eigvec[:,self.eom_eigval_argsort[i]]
                s = round(2*(-1+np.sqrt(1+4*s2_val)))
                s /= 4
                print(f'{i:^4d} {self.eom_eigval[self.eom_eigval_argsort[i]]:^25.12f} {abs(s2_val):^10.3f} {abs(s):^5.1f}')
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
             
            
if __name__ == "__main__":
    test = 1
    if (test == 1): 
        psi4.core.set_output_file('beh_output.dat', False)
        bohr = 0.529177210903
        r = 0.7
        y = bohr * r
        m = 4.0/(0.7 - 2.54)
        b = -2.54 * m * bohr
        z = (m * y) + b
        mol = psi4.geometry(f"""
        Be 0.0   0.0             0.0
        H  0.0   {y}   {z}
        H  0  -{y}   {z}
        symmetry c2v
        """)

        psi4.set_options({
            'basis': 'sto-6g',
            'frozen_docc': [0,0,0,0],
            'restricted_docc':[2,0,0,0],
            'reference': 'rhf',
        })

        forte_options = {
            'basis': 'sto-6g',
            'job_type': 'mcscf_two_step',
            'active_space_solver': 'fci',
            'frozen_docc': [0,0,0,0],
            'restricted_docc':[2,0,0,0],
            'active':[1,0,0,1],
            'root_sym': 0,
            'maxiter': 100,
            'e_convergence': 1e-8,
            'r_convergence': 1e-8,
            'casscf_e_convergence': 1e-8,
            'casscf_g_convergence': 1e-6,
        }

        E_casscf, wfn_cas = psi4.energy('forte', forte_options=forte_options, return_wfn=True)

        print(f'CASSCF Energy = {E_casscf}')

        mos_spaces = {'GAS1' : [2,0,0,0], 
                    'GAS2' : [1,0,0,1],
                    'GAS3' : [1,0,1,1]
                    }
        # active_docc = [0] means that the first orbital in the active space is doubly occupied.
        eomcc = EOM_CC(mos_spaces, wfn_cas, verbose=True, active_docc = [0]) # Closed-shell singlet is assumed.
        eomcc.make_T()
        eomcc.run_ccn()
        eomcc.run_eom(sym = 0) # The target symmetry for excited states.