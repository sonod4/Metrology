from metrology import Analyzer
from torch import matrix_exp,tensor,cos,sin,bmm
import numpy as np
from numpy import sqrt,pi
from time import time

np.random.seed(0)

class ScramblingModel:
	def __init__(self,n,gamma,representationDimension):
		"""
		Model implementing the scrambling model. Parameters are in the form
		of (lambda_1,lambda_2)
		IMPORTANT: Generators i.e. spin 1/2 are J_i = 1/2 sigma_i, be careful
					to match the choice convention.

		Parameters:
		n (np.ndarray): Direction of the scrambling hamiltonian.
		gamma (float): Intensity of the scrambling hamiltonian.
		representationDimension (int): Dimension of the representation matrix (representation = 2j+1, with j the spin).
		"""
		# Mandatory parameters
		self.dH = representationDimension
		self.npars = 2

		self.n = tensor(n)
		self.gamma = tensor(gamma)

		# Pauli Matrices
		self.PM = self.findGenerators(representationDimension)

	def __call__(self, pars):
		return bmm(matrix_exp(-1j*pars[:,1][:,None,None]*self.PM[2]),
					bmm(matrix_exp(-1j * self.gamma * (  self.n[0]*self.PM[0] + self.n[1]*self.PM[1] + self.n[2]*self.PM[2]  )).expand(pars.shape[0], -1, -1),
						matrix_exp(-1j*pars[:,0][:,None,None]*self.PM[2])  ))

	def findGenerators(self, representationDimension):
		# spin j
		j = (representationDimension-1)/2
		m = [-j+i for i in range(0,representationDimension)]

		J_z = np.diag(m[::-1]).astype(np.complex128)

		J_plus = np.zeros((representationDimension,representationDimension),dtype=np.complex128)
		for i in range(len(m)-1):
			# int = floor
			J_plus[i][i+1] = np.sqrt((j-m[i])*(j+m[i]+1))

		J_minus = J_plus.T

		J_x = 0.5*(J_plus+J_minus)
		J_y = -0.5j*(J_plus-J_minus)

		if False:
			print(J_x,J_y,J_z)
			print("Check commutation relations: ")
			print(J_x@J_y-J_y@J_x -1j*J_z)
			print(J_x@J_z-J_z@J_x +1j*J_y)
			print(J_z@J_y-J_y@J_z +1j*J_x)

		return tensor([J_x,J_y,J_z])



# ========= Model creation =========
# fixed parameters
theta = 0.6
phi = 0.2
n = np.array([np.cos(theta),np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)])
gamma = 1.0

dim = 2
scrambler = ScramblingModel(n,gamma,representationDimension=dim)
model = Analyzer(model=scrambler)

# ========= Statistical properties =========
t = time()
Nstates = 1_000
Npars = 1_000
print("Starting analysis:")
metrics = ["s","c","R","T","C_SLD","C_W","Delta_R_T"]
r = model.fastSampleAnalysis(Nstates=Nstates,Npars=Npars,parametersRange=np.array([[0,2*pi],[0,2*pi]]),metrics=metrics,
							save=f"data/scrambler_{dim}bit_{Nstates}_{Npars}.pkl")
print("Calculation time: ", time()-t)



