"""This class is to find lowest eigenvalues with Davidson-Liu algorithm."""

import logging
import warnings

import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


class Davidson(object):
    """Davidson-Liu algorithm to get the n states with smallest eigenvalues."""
    
    def __init__(self, matrix, max_subspace = 100, max_iterations = 300, eps = 1e-6):
        self.matrix = matrix
        self.diagonal = matrix.diagonal()
        self.max_subspace = max_subspace
        self.max_iterations = max_iterations
        self.eps = eps
        
    def kernal(self, n_lowest=1, initial_guess = None):
        # Checks for number of states desired, should be in the range of
        # [0, max_subspace).
        if n_lowest <= 0 or n_lowest >= self.max_subspace:
            raise ValueError(
                'n_lowest {} is supposed to be in [1, {}).'.format(
                    n_lowest, self.max_subspace
                )
            )
        
        # Checks for the initial guess.    
        if initial_guess is None:
            initial_guess = np.zeros((len(self.diagonal), n_lowest))
            np.fill_diagonal(initial_guess, 1)
            
        sucess = False
        niter = 0
        guess_v = initial_guess
        
        while niter < self.max_iterations and not sucess:   
            guess_mv = np.einsum('nm, mj->nj', self.matrix, guess_v, optimize='optimal')
            guess_vmv = np.einsum('ni,nj->ij', guess_v, guess_mv, optimize='optimal')
            trial_lambda, trial_transformation = np.linalg.eigh(guess_vmv)
            
            # 1. Sorts eigenvalues in ascending order.
            # sorted_index = list(reversed(trial_lambda.argsort()[::-1]))
            # trial_lambda = trial_lambda[sorted_index]
            # trial_transformation = trial_transformation[:, sorted_index]

            if len(trial_lambda) > n_lowest:
                trial_lambda = trial_lambda[:n_lowest]
                trial_transformation = trial_transformation[:, :n_lowest]

            # 2. Estimates errors based on diagonalization in the smaller space.
            trial_v = np.dot(guess_v, trial_transformation) # Guess eigenvectors in the original space.
            trial_mv = np.einsum('nm, mj->nj', self.matrix, trial_v, optimize='optimal') # Guess Ax in the original space.
            trial_error = trial_mv - trial_v * trial_lambda # Residual vectors in the original space.
            
            # 3. Gets new directions from error vectors.
            max_error = 0 # Maximum error for the current iteration.
            new_directions = []
            full_dim = trial_v.shape[0]
            for i in range(n_lowest):
                current_error_v = trial_error[:, i]
                if np.max(np.abs(current_error_v)) < self.eps:
                    continue
                max_error = max(max_error, np.linalg.norm(current_error_v))
                
                new_direction = []
                M = np.ones(full_dim)
                for j in range(full_dim):
                    diff = self.diagonal[j] - trial_lambda[i]        
                    if numpy.abs(diff) > self.eps:
                        M[j] /= diff
                    else:
                        M[j] /= self.eps  
                    
                numerator = M * current_error_v 
                denominator = M * trial_v[:, i]
                new_direction = -current_error_v + (trial_v[:, i] * numpy.dot(trial_v[:, i], numerator) / numpy.dot(trial_v[:, i], denominator)) # This line is fine.
                # new_direction *= M
                
                # new_direction = M * current_error_v # This is the Davidson-Liu way.
                
                # P = np.identity(full_dim) - np.outer(trial_v[:, i], trial_v[:, i]) 
                # A_tilde = np.diagonal(self.matrix) - np.identity(full_dim) * trial_lambda[i]
                # M_tilde = np.einsum('ij, jk, kl->il', P, A_tilde, P, optimize='optimal')
                # new_direction, _ = scipy.sparse.linalg.cg(M_tilde, -current_error_v, atol = self.eps, maxiter = self.max_iterations) # This is the Jacobi-Davidson way.
                
                new_directions.append(new_direction)
                
                
            if new_directions:
                # stack new_directions along the axis 0, then transpose, finally hstack.
                guess_v = np.hstack([guess_v, np.stack(new_directions).T]) 
                
            print(
                f"Eigenvalues for iteration {niter}: {trial_lambda}, error is {max_error}."
            )
            
            if max_error < self.eps:
                success = True
                break
        
            # 4. Deals with new directions to make sure they're orthonormal.
            ortho_num = guess_mv.shape[1] # Already orthonormal
            for i in range(ortho_num, guess_v.shape[1]):
                vec_i = guess_v[:, i]
                for j in range(i):
                    vec_i -= guess_v[:, j] * np.dot(guess_v[:, j], vec_i)
                    
                # Makes sure the new vector is not too small.
                if np.linalg.norm(vec_i) < self.eps:
                    continue
                
                guess_v[:, i] = vec_i / np.linalg.norm(vec_i)
                ortho_num += 1
                
            # 5. Limits the size of the subspace.
            if guess_v.shape[1] > self.max_subspace:
                print("Collapsing the subspace.")
                guess_v = trial_v
                guess_mv = trial_mv
                
            
            niter += 1
            
        return trial_lambda, trial_v
                    
if (__name__=='__main__'):
    np.random.seed(100)
    A = np.random.rand(500, 500)
    A = A + A.T
    
    davidson = Davidson(A)
    trial_lambda, trial_v = davidson.kernal(n_lowest=2)    
    print(f"Eigenvalues from Davidson: {trial_lambda}")    
    evals, evecs = np.linalg.eigh(A)
    print(f"Eigenvalues from Numpy: {evals[:2]}")    
            
        
                

