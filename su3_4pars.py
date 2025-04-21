from metrology import Analyzer
from torch import matrix_exp,tensor,cos,sin
from torch import complex128 as cp128
from time import time
from numpy import sqrt,pi
import numpy as np

np.random.seed(0)


class SU3Model:
	def __init__(self,time,representationDimension):
		"""
		Model implementing a unitary SU(2) evolution over a nbit. Parameters are in the
		form of (B,theta)
		IMPORTANT: Generators i.e. spin 1/2 are J_i = 1/2 sigma_i, be careful
					to match the choice convention.

		Parameters:
		time (float): Duration of application of the evolution.
		representationDimension (int): Dimension of the representation matrix (representation = 2j+1, with j the spin).
		"""
		self.dH = representationDimension
		self.npars = 4

		self.time = time

		self.J = self.findGenerators(self.dH)

	def __call__(self, pars):
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/abfb8f5b7936e79c6e259326a3869aab391c3c8c
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/e4a805778bf9d799471fa3a5c630f10ede8cdb65
		# 3 pars
		# return matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*cos(pars[:,2])[:,None,None]*self.J[0] + cos(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*self.J[1] + sin(pars[:,1])[:,None,None]*self.J[2] ) )
		
		# 4 pars
		return matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*cos(pars[:,2])[:,None,None]*self.J[1] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*cos(pars[:,3])[:,None,None]*self.J[2] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*self.J[3] ) )
		
		# 5 pars
		# return matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*cos(pars[:,2])[:,None,None]*self.J[1] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*cos(pars[:,3])[:,None,None]*self.J[2] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*cos(pars[:,4])[:,None,None]*self.J[3] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*sin(pars[:,4])[:,None,None]*self.J[4] ) )
		
		# 6 pars
		# return matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*cos(pars[:,2])[:,None,None]*self.J[1] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*cos(pars[:,3])[:,None,None]*self.J[2] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*cos(pars[:,4])[:,None,None]*self.J[3] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*sin(pars[:,4])[:,None,None]*cos(pars[:,5])[:,None,None]*self.J[4] + sin(pars[:,1])[:,None,None]*sin(pars[:,2])[:,None,None]*sin(pars[:,3])[:,None,None]*sin(pars[:,4])[:,None,None]*sin(pars[:,5])[:,None,None]*self.J[5] ) )

	def findGenerators(self, representationDimension):
		# https://it.wikipedia.org/wiki/Matrici_di_Gell-Mann
		if representationDimension == 3:
			# spin 1
			J_1 = tensor([[0,1,0],[1,0,0],[0,0,0]], dtype=cp128)
			J_2 = tensor([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=cp128)
			J_3 = tensor([[1,0,0],[0,-1,0],[0,0,0]], dtype=cp128)
			J_4 = tensor([[0,0,1],[0,0,0],[1,0,0]], dtype=cp128)
			J_5 = tensor([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=cp128)
			J_6 = tensor([[0,0,0],[0,0,1],[0,1,0]], dtype=cp128)
			J_7 = tensor([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=cp128)
			J_8 = tensor([[1/sqrt(3),0,0],[0,1/sqrt(3),0],[0,0,-2/sqrt(3)]], dtype=cp128)

		return [J_1,J_2,J_3,J_4]


dim = 3
suppression = [0,0]
su3 = SU3Model(time=1.0,representationDimension=dim)
model = Analyzer(model=su3,suppress=suppression)

# ========= Statistical properties =========
t = time()
Nstates = 1_000
Npars = 1_000
print("Starting analysis:")
metrics = ["s","c","R","T","C_SLD","C_W","Delta_R_T"]
r = model.fastSampleAnalysis(Nstates=Nstates,Npars=Npars,parametersRange=np.array([[0,2*pi],[0,2*pi],[0,2*pi],[0,2*pi]]), metrics=metrics,
							save=f"data/su3_4pars_{dim}bit_{suppression[0]},{suppression[1]}suppress_{Nstates}_{Npars}.pkl")
print("Calculation time: ", time()-t)


