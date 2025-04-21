from metrology import Analyzer
from torch import matrix_exp,tensor,cos,sin,einsum,kron
from time import time
from numpy import sqrt,pi
import numpy as np

np.random.seed(0)

class SU2Model:
	def __init__(self,time,representationDimension,dK,entanglement_n):
		"""
		Model implementing a unitary SU(2) evolution over a nbit. Parameters are in the
		form of (B,theta)
		IMPORTANT: Generators i.e. spin 1/2 are J_i = 1/2 sigma_i, be careful
					to match the choice convention.

		Parameters:
		time (float): Duration of application of the evolution.
		representationDimension (int): Dimension of the representation matrix (representation = 2j+1, with j the spin).
		"""
		# Mandatory parameters
		self.dH = representationDimension
		self.npars = 2

		# Fixed parameter
		self.time = time
		self.dK = dK
		self.entanglement_n = entanglement_n

		# Generators
		self.J = self.findGenerators(self.dH)

	def __call__(self, pars):
		U = matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*self.J[2] ) )
		
		# U' = U \otimes 1
		if True:
			I_k = tensor(np.eye(self.dK))
			return kron(U, I_k)

		# U" = U \otimes U
		if False:
			kron_intermediate = torch.einsum('nij,nkl->nikjl', U, U)
			result = kron_intermediate.reshape(U.shape[0], U.shape[1] * U.shape[1], U.shape[2] * U.shape[2])
			return result

		raise Exception("No evolution selected")

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


dim = 2
dim2 = 2
n = 3
suppression = [0,0]
su2 = SU2Model(time=1.0,representationDimension=dim,dK=dim2,entanglement_n=n)
# a = tensor([[1.1,2.2],[3.3,4.4]])
# print(su2(a))
# exit()
model = Analyzer(model=su2,suppress=suppression)

# ========= Statistical properties =========
t = time()
Nstates = 1_000
Npars = 1_000
print("Starting analysis:")
metrics = ["s","c","R","T","C_SLD","C_W","Delta_R_T"]
r = model.fastSampleAnalysis(Nstates=Nstates,Npars=Npars,parametersRange=np.array([[0,2*pi],[0,2*pi]]), metrics=metrics,
							save=f"data/UI_{n}ENT_su2_2pars_{dim}bit_{Nstates}_{Npars}.pkl")
print("Calculation time: ", time()-t)




