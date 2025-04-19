import numpy as np
from numpy import pi,sqrt


# ================================================================================================= #
# ============================================ SAMPLER ============================================ #
# ================================================================================================= #

class Sampler:
	def __init__(self, model, suppress=[0,0]):
		"""
		Sampler, that given a certain model, and a certain finite Hilbert space,
		it samples uniformly over the Hilbert space.

		Parameters:
		model (function or class): User defined function that returns the evolution unitary.
							Must take 1 obligatory parameter, a np.ndarray of parameters
		suppress (list): Number of rows from the end to set to 0, and number of additional phases
							to set to 0.
		"""
		self.model = model
		self.suppress = suppress
		if self.suppress[0]>self.model.dH-1:
			raise Exception(f"Error: trying to suppress {self.suppress[0]} rows of qudit, when only {self.model.dH} rows are available(and 1 must be non zero).")
		if self.suppress[0]+self.suppress[1]>self.model.dH-1:
			raise Exception(f"Error: trying to suppress phases, but {self.suppress[0]} rows of the qudit have been removed({self.model.dH-self.suppress[0]} left), meaning only up to {self.model.dH-1-self.suppress[0]} phases remains, and can therefore be suppressed({self.suppress[1]} phases are too many).")

	def samplePureStates(self,dim,N=1):
		"""Sample a generic state with uniform probability,
		the algorithm samples states in a (0,1)^dH box, then
		only states inside the R=1 dHsphere are kept, and
		renormalized into its surface.

		Parameters:
		dim (int): Dimension of the Hilbert space to sample on.
		N (int): number of states to be sampled

		Returns:
		numpy array of elements of form (a,b,...,phase_b,phase_c,...)
		"""
		n = 100_000 # instead of doing the whole thing it's done in batches

		# sampling coefficients (a,b,c,d)
		numbers = np.random.uniform(0, 1, (min(10*N,n),dim))
		coefficients = numbers[np.sum(numbers**2,axis=1)<1] # selected points withing the sphere
		coefficients = coefficients / sqrt(np.sum(coefficients**2,axis=1, keepdims=True)) # normalized
		while len(coefficients)<N:
			numbers = np.random.uniform(0, 1, (min(10*N,n),dim))
			numbers = numbers[np.sum(numbers**2,axis=1)<1]
			numbers = numbers / sqrt(np.sum(numbers**2,axis=1, keepdims=True))
			coefficients = np.concatenate((coefficients,numbers))

		# sampling phases
		phases = np.random.uniform(0, 2*pi, (N,dim-1))

		for i in range(self.suppress[0]):
			coefficients[:,dim-1-i] = coefficients[:,dim-1-i]*0
		if self.suppress[0]>0:
			coefficients = coefficients / sqrt(np.sum(coefficients**2,axis=1, keepdims=True))

		for i in range(self.suppress[1]):
			phases[:,dim-2-self.suppress[0]-i] = phases[:,dim-2-self.suppress[0]-i]*0

		return np.concatenate((coefficients[:N],phases),axis=1)

	def constructStatisticalOperator(self, states, dim):
		# states of the form (a,b,c,...,phi_b,phi_c,...)
		psi = self.buildKet(states, dim)

		return np.conj(psi[:,:,None])*psi[:,None,:] # perform outer product on each state

	def sampleStatisticalOperators(self, N=1):
		states = self.samplePureStates(self.model.dH, N)
		return self.constructStatisticalOperator(states, self.model.dH)

	def sampleParameters(self, ranges, N=1):
		return np.column_stack( [np.random.uniform(ranges[i][0], ranges[i][1], (N,1)) for i in range(len(ranges))] )

	def buildKet(self, states, dim):
		psi = states[:,:dim].astype(np.complex128)
		phases = np.exp(1j * states[:,dim:])
		phases = np.hstack((np.ones((states.shape[0], 1)), phases),dtype=np.complex128)
		psi = psi*phases
		return psi

	def sampleEntangledStatisticalOperators(self, N=1):
		if not hasattr(self.model,"entanglement_n"):
			raise Exception('Missing "entanglement_n" variable definition in Model passed.')
		# N is the number of this kind of states to sample
		psi = self.samplePureStates(self.model.dH, self.model.entanglement_n*N)
		phi = self.samplePureStates(self.model.dK, self.model.entanglement_n*N)

		psi = self.buildKet(psi,self.model.dH)
		phi = self.buildKet(phi,self.model.dK)
		
		Psi = np.einsum('ni,nj->nij', psi, phi).reshape(self.model.entanglement_n*N, self.dH*self.dK)
		Psi = Psi.reshape(N, self.model.entanglement_n, self.dH*self.dK).sum(axis=1)
		Psi = Psi / np.sqrt(np.sum(Psi.conj()*Psi,axis=1, keepdims=True))

		rho = np.conj(Psi[:,:,None])*Psi[:,None,:]
		return rho

