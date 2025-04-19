from .sampler import *
import numpy as np
from numpy import e,pi,sqrt
from time import time
from threading import Thread
from torch import tensor,matmul,eye
from torch import complex128 as cp128
from numba import njit,prange,complex128,int64,boolean
import pickle as pkl
from scipy.linalg import sqrtm

# ================================================================================================= #
# =========================================== ANALYZER ============================================ #
# ================================================================================================= #

@njit(parallel=True,cache=True)
def fastAnalyze(final_state, L, sqrtW, W, Nstates, Npars, npars, metrics):
	# # matrices of relative metrics
	if "s" in metrics:				s = np.zeros((Nstates,Npars),dtype=np.float64)
	if "c" in metrics:				c = np.zeros((Nstates,Npars),dtype=np.float64)
	if "R" in metrics:				R = np.zeros((Nstates,Npars),dtype=np.float64)
	if "T" in metrics:				T = np.zeros((Nstates,Npars),dtype=np.float64)
	if "C_SLD" in metrics:			C_SLD = np.zeros((Nstates,Npars),dtype=np.float64)
	# if "C_R" in metrics:			C_R = np.zeros((Nstates,Npars),dtype=np.float64)
	# if "C_H"  in metrics:			C_H = np.zeros((Nstates,Npars),dtype=np.float64)
	if "C_W"  in metrics:			C_W = np.zeros((Nstates,Npars),dtype=np.float64)
	# if "Delta_H_SLD" in metrics:	Delta_H_SLD = np.zeros((Nstates,Npars),dtype=np.float64)
	if "Delta_R_T" in metrics:		Delta_R_T = np.zeros((Nstates,Npars),dtype=np.float64)
	# if "Delta_R_H" in metrics:		Delta_R_H = np.zeros((Nstates,Npars),dtype=np.float64)
	# if "Delta_T_H" in metrics:		Delta_T_H = np.zeros((Nstates,Npars),dtype=np.float64)

	for i in prange(Nstates):
		for j in prange(Npars):
			G_ = np.zeros((npars,npars),dtype=np.complex128)
			for k in range(npars):
				for z in range(npars):
					G_[k][z] = np.trace( np.dot(final_state[i][j],np.dot(L[i,j*npars+k],L[i,j*npars+z])) )
			# _ used for variables in the specific sampling
			Q_ = np.real(G_).astype(np.float64)
			D_ = np.imag(G_).astype(np.float64)


			# useful quantities
			QI_ = np.linalg.inv(Q_).astype(np.float64)
			QID_ = np.dot(QI_,D_).astype(np.float64)
			QIDQI_ = np.dot(QID_,QI_).astype(np.float64)
			norm1_WQIDQIW_ = np.linalg.norm(np.dot(sqrtW,np.dot(QIDQI_,sqrtW)),ord=1)#.astype(np.float64)
			# norm1_WQIDQIW_ = np.linalg.norm(QIDQI_,ord=1)

			C_SLD_ = np.trace(np.dot(W,QI_))
			R_ = np.abs(np.linalg.eigvals( QID_.astype(np.complex128) )).max() # norm infinity
			T_ = norm1_WQIDQIW_/C_SLD_

			if "s" in metrics:				s[i][j] = 1/np.linalg.det(Q_)
			if "c" in metrics:				c[i][j] = 2/np.trace( np.dot(np.conj(D_).T,D_) )
			if "R" in metrics:				R[i][j] = R_
			if "T" in metrics:				T[i][j] = T_
			if "C_SLD" in metrics:			C_SLD[i][j] = C_SLD_
			# if "C_R" in metrics:			C_R[i][j] = C_SLD_+?
			# if "C_H"  in metrics:			C_H[i][j] = C_SLD_+norm1_WQIDQIW_ # PURE MODEL ASSUMPTION
			if "C_W"  in metrics:			C_W[i][j] = C_SLD_+norm1_WQIDQIW_
			# if "Delta_H_SLD" in metrics:	Delta_H_SLD[i][j] = norm1_QIDQI_/C_SLD_
			if "Delta_R_T" in metrics:		Delta_R_T[i][j] = R_ - T_
			# if "Delta_R_H" in metrics:		Delta_R_H[i][j] = R_ - norm1_QIDQI_/C_SLD_
			# if "Delta_T_H" in metrics:		Delta_T_H[i][j] = T_ - norm1_QIDQI_/C_SLD_

	r = {}
	if "s" in metrics:				r["s"] = s
	if "c" in metrics:				r["c"] = c
	if "R" in metrics:				r["R"] = R
	if "T" in metrics:				r["T"] = T
	if "C_SLD" in metrics:			r["C_SLD"] = C_SLD
	# if "C_R" in metrics:			r["C_R"] = C_R
	# if "C_H"  in metrics:			r["C_H"] = C_H
	if "C_W"  in metrics:			r["C_W"] = C_W
	# if "Delta_H_SLD" in metrics:	r["Delta_H_SLD"] = Delta_H_SLD
	if "Delta_R_T" in metrics:		r["Delta_R_T"] = Delta_R_T
	# if "Delta_R_H" in metrics:		r["Delta_R_H"] = Delta_R_H
	# if "Delta_T_H" in metrics:		r["Delta_T_H"] = Delta_T_H
	return r



class Analyzer:
	def __init__(self, model, suppress=[0,0]):
		"""
		Unitary model describing the unitary evolution, parametrized
		by a vector of npars parameters, which are to be estimated.
		Constant, or additional parameters that aren't of interest
		can be passed in the model.

		Parameters:
		model (function or class): User defined function that returns the evolution unitary.
							Must take 1 obligatory parameter, a np.ndarray of parameters
		suppress (list): Number of rows from the end to set to 0, and number of additional phases
							to set to 0.
		"""
		self.suppress = suppress
		self.sampler = Sampler(model=model, suppress=suppress)
		self.available_metrics = ["s","c","R","T","C_SLD","C_W","Delta_R_T"]
		self.DEFAULT_EPSILON = 0.00000001

	def __call__(self, pars):
		return self.sampler.model(pars)

	# ============================================= ANALYZER FUNCTIONS ============================================= #

	def fastSampleAnalysis(self, state=None, pars=None, parametersRange=None, Nstates=1,Npars=1, metrics=[], W=None, epsilon=None,
							save=None, parallelize=True, verbose=True):
		if epsilon is None: epsilon=self.DEFAULT_EPSILON
		"""
		Analysis of the model sampling over N states, valued at
		given parameters.

		Parameters:
		state (np.ndarray): Statistical operator state to evaluate the model on.
		pars (np.ndarray): Vector of parameters of the given model.
		parametersRange (np.ndarray): Vector of ranges onto which sample parameters,
								i.e. np.array([ [0,2*pi],[0,2*pi] ])
		Nstates (int): Number of sample states to use for the statistical analysis.
		Npars (int): Number of sample parameters to use for the statistical analysis.
		metrics (lists): List of strings, which specify the metrics to keep track of.
		W (np.ndarray): Weight matrix, default value is the identity.
		epsilon (float): Specify the step increment in doing derivatives in the incremental way.
		save (string): If not None, it's the name of the file with the dictionary.
		parallelize (bool): Flag to parallelize calculation of the SLDs.
		verbose (bool): If True time checkpoints are reported.

		Returns:
		Values of specified metrics for the Nstates*Npars sampled values.
		"""
		if pars is None and parametersRange is None:
			raise Exception("If a set of parameters is not passed, a parameters range MUST be passed.")

		# if singular state of shape (dH,dH) is passed, it's reshaped into (1,dH,dH)
		if not(state is None) and len(state.shape)==2: state = state.reshape(1,state.shape[0],state.shape[1]) 
		if not(state is None) and len(state.shape)==3:
			raise Exception("Input shape of states must be either (dH,dH), or (Nstates,dH,dH).")
		# if singular parameters set of shape (npars) is passed, it's reshaped into (1,npars)
		if not(pars is None) and len(pars.shape)==1: pars = pars.reshape(1,pars.shape[0]) 
		if not(pars is None) and len(pars.shape)==2:
			raise Exception("Input shape of parameters must be either (npars), or (Npars,npars).")

		for m in metrics:
			if m not in self.available_metrics:
				raise Exception(f"Invalid metric: available metrics to calcualte are: {self.available_metrics}")

		# sampling states and or parameters
		if state is None:
			if hasattr(self.sampler.model,"dK"):
				if self.sampler.model.dK != 0:
					state = self.sampler.sampleEntangledStatisticalOperator(N=Nstates)
			else:
				state = self.sampler.sampleStatisticalOperators(N=Nstates)
		if pars is None:
			pars = self.sampler.sampleParameters(ranges=parametersRange,N=Npars)

		if W is None:
			W = np.diag([1]*self.sampler.model.npars)
		elif len(W.shape)!=2 or W.shape[0]!=2 or W.shape[1]!=2:
			raise Exception("Input shape for W must be (npars,npars).")

		t = time()
		# Analyzing for each combination of (state,parameters)
		Nstates = state.shape[0]
		Npars = pars.shape[0]
		L = [0]*Nstates
		final_state = [0]*Nstates
		t = time()

		if parallelize:
			threads = []
			for i in range(Nstates):
				thread = Thread(target=self.fastSLD, args=(state[i],pars,L,final_state,i,epsilon))
				threads.append(thread)
				thread.start()
			for thread in threads:
				thread.join()
		else:
			# non parallelized way, slightly slower
			for i in range(Nstates):
				self.fastSLD(input_state=state[i],pars=pars,LL=L,FINAL_STATE=final_state,ii=i,epsilon=epsilon)

		L = np.array(L)
		final_state = np.array(final_state)

		if verbose:
			print("Calculated SLDs in: ",time()-t)
			print("Calculating metrics")

		t = time()
		r = fastAnalyze(final_state,L,sqrtm(W).astype(np.float64),W.astype(np.float64),Nstates,Npars,self.sampler.model.npars,metrics)
		if verbose:
			print("Done in: ",time()-t)

		r = dict(r)
		if not(save is None):
			self.save(save, r)
		return r

	def save(self, filename, obj):
		with open(filename, 'wb') as file:
			pkl.dump(obj,file)

	# ============================================= SLD calculation ============================================= #

	def fastSLD(self, input_state, pars, LL, FINAL_STATE, ii, epsilon=None):
		# Important: takes in input shape (N,npars)
		if epsilon is None: epsilon=self.DEFAULT_EPSILON

		# Ensure torch tensors
		pars = tensor(pars, dtype=cp128)
		input_state = tensor(input_state, dtype=cp128)
		delta = eye(self.sampler.model.npars, dtype=pars.dtype, device=pars.device) * epsilon
		# Expand batch: (npars, N, npars)
		pars_expanded = pars.unsqueeze(0).expand(self.sampler.model.npars, pars.shape[0], self.sampler.model.npars)
		# Broadcast deltas over batch: (npars, 1, npars) -> (npars, N, npars)
		delta_expanded = delta.unsqueeze(1)
		# Perturbed parameters
		perturbed_pars = pars_expanded + delta_expanded  # shape: (npars, N, npars)

		perturbed_pars = perturbed_pars.reshape(-1, 2)

		perturbed_pars = (pars_expanded + delta_expanded).permute(1, 0, 2).reshape(-1, self.sampler.model.npars)
		U = self(pars)
		Udagger = U.conj().transpose(-2, -1)
		pU = self(perturbed_pars) # (N*npars,dH,dH)
		pUdagger = pU.conj().transpose(-2, -1)

		final_state = matmul(U, matmul(input_state, Udagger))
		perturbed_final_state = matmul(pU, matmul(input_state, pUdagger))
		L = 2*(perturbed_final_state-final_state.repeat_interleave(self.sampler.model.npars, dim=0))/epsilon

		LL[ii] = L.cpu().detach().numpy()
		FINAL_STATE[ii] = final_state.cpu().detach().numpy()
		# return L.cpu().detach().numpy(), final_state.cpu().detach().numpy()

