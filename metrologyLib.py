import numpy as np
from numpy import e,pi,cos,sin,sqrt
from time import time
from scipy.linalg import expm
import matplotlib.pyplot as plt
from threading import Thread

class Model:
	def __init__(self,model,npars,dH):
		"""
		Unitary model describing the unitary evolution, parametrized
		by a vector of npars parameters, which are to be estimated.
		Constant, or additional parameters that aren't of interest
		can be passed in the model.
		Important: analysis can be conducted on single setups, it's not
		optimized to numpy broadcast multiple analysis in parallel.

		Parameters:
		model (function or class): User defined function that returns the evolution unitary.
							Must take 1 obligatory parameter, a np.ndarray of parameters
		npars (int): Number of parameters to be estimated.
		dH (int): Dimension of the Hilbert space.
		"""
		self.model = model
		self.sampler = Sampler(npars=npars,dH=dH)
		self.npars = npars
		self.dH = dH
		self.DEFAULT_EPSILON = 0.00001

	def __call__(self, pars):
		return self.model(pars)

	def sampleAnalysis(self, sampleOverStates, sampleOverPars, state=None, pars=None, parametersRange=None,
						Nstates=1,Npars=1, s=False, c=False, R=False, Q=False, D=False, C_SLD=False, C_H=False, Delta=False,
						epsilon=None, PARALLELIZE=False):
		# if epsilon is None: epsilon=self.DEFAULT_EPSILON
		"""
		Analysis of the model sampling over N states, valued at
		given parameters.

		Parameters:
		sampleOverStates (bool): If True a sampling over states is done.
		sampleOverPars (bool): If True a sampling over parameters is done.
		state (np.ndarray): Statistical operator state to evaluate the model on.
		pars (np.ndarray): Vector of parameters of the given model.
		parametersRange (np.ndarray): Vector of ranges onto which sample parameters,
								i.e. np.array([ [0,2*pi],[0,2*pi] ])
		Nstates (int): Number of sample states to use for the statistical analysis.
		Npars (int): Number of sample parameters to use for the statistical analysis.
		s,c,R,Q,D,C_SLD,C_H,Delta (bool): If True they are calculated and returned.
		epsilon (float): Specify the step increment in doing derivatives in the incremental way.
		PARALLELIZE (bool): Flag for parallelization of the independent sampling settings calculation.

		Returns:
		Values of specified metrics for the N sampled values.
		"""
		if sampleOverStates and not(state is None):
			raise Exception("If sampling over states, no state has to be passed.")
		if sampleOverPars and not(pars is None):
			raise Exception("If sampling over parameters, no parameters have to be passed.")
		if sampleOverPars and parametersRange is None:
			raise Exception("If sampling over parameters, the range of interest MUST be passed.")
		if sampleOverPars==False and pars is None and parametersRange is None:
			raise Exception("If a set of parameters is not passed, a parameters range MUST be passed.")

		# if singular state of shape (dH,dH) is passed, it's reshaped into (1,dH,dH)
		if sampleOverStates==False and not(state is None) and len(state.shape)!=3: state = state.reshape(1,state.shape[0],state.shape[1]) 
		# if singular parameters set of shape (npars) is passed, it's reshaped into (1,npars)
		if sampleOverPars==False and not(pars is None) and len(pars.shape)!=2: pars = pars.reshape(1,pars.shape[0]) 

		# matrices of relative metrics
		s_,c_,R_,Q_,D_ = None,None,None,None,None
		if s: s_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if c: c_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if R: R_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if Q: Q_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if D: D_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if C_SLD: C_SLD_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if C_H: C_H_ = np.zeros((Nstates,Npars),dtype=np.float64)
		if Delta: Delta_ = np.zeros((Nstates,Npars),dtype=np.float64)

		# sampling states and or parameters
		if sampleOverStates:
			state = self.sampler.sampleStatisticalOperator(N=Nstates)
		if sampleOverStates==False and state is None:
			state = self.sampler.sampleStatisticalOperator(N=1)
		if sampleOverPars:
			pars = self.sampler.sampleParameters(ranges=parametersRange,N=Npars)
		if sampleOverPars==False and pars is None:
			pars = self.sampler.sampleParameters(ranges=parametersRange,N=1)

		# calculating each combination
		Nstates = state.shape[0]
		Npars = pars.shape[0]
		if PARALLELIZE:
			def calculateij(self,i,j,s,c,R,Q,D,s_,c_,R_,Q_,D_,Npars):
				Q__,D__,s__,c__,R__,C_SLD__,C_H__,Delta__ = self.analyzeState(input_state=state[i],pars=pars[j])
				if s: s_[i][j] = s__
				if c: c_[i][j] = c__
				if R: R_[i][j] = R__
				if Q: Q_[i][j] = Q__
				if D: D_[i][j] = D__
				if C_SLD: C_SLD_[i][j] = C_SLD__
				if C_H: C_H_[i][j] = C_H__
				if Delta: Delta_[i][j] = Delta__
			threads = []
			for i in range(Nstates):
				for j in range(Npars):
					thread = Thread(target=calculateij, args=(self,i,j,s,c,R,Q,D,s_,c_,R_,Q_,D_,Npars))
					threads.append(thread)
					thread.start()
			for thread in threads:
				thread.join()
		else:
			for i in range(Nstates):
				for j in range(Npars):
					Q__,D__,s__,c__,R__,C_SLD__,C_H__,Delta__ = self.analyzeState(input_state=state[i],pars=pars[j])
					if s: s_[i][j] = s__
					if c: c_[i][j] = c__
					if R: R_[i][j] = R__
					if Q: Q_[i][j] = Q__
					if D: D_[i][j] = D__
					if C_SLD: C_SLD_[i][j] = C_SLD__
					if C_H: C_H_[i][j] = C_H__
					if Delta: Delta_[i][j] = Delta__
		# result dictionary
		r = {}
		r["states"] = state
		r["parameters"] = pars
		if s: r["s"] = s_
		if c: r["c"] = c_
		if R: r["R"] = R_
		if Q: r["Q"] = Q_
		if D: r["D"] = D_
		if C_SLD: r["C_SLD"] = C_SLD_
		if C_H: r["C_H"] = C_H_
		if Delta: r["Delta"] = Delta_

		return r

	def analyzeState(self, input_state, pars, epsilon=None):
		if epsilon is None: epsilon=self.DEFAULT_EPSILON
		"""
		Analysis of the model with given input state, valued at
		given parameters.

		Parameters:
		input_state (np.ndarray,complex128): Density matrix.
		pars (np.ndarray,complex128): Vector of parameters of the given model.
		epsilon (float): Specify the step increment in doing derivatives in the incremental way.

		Returns:
		QFIM, mean Uhlmann curvature, sloppiness, compatibility, asimptotic incompatibility
		"""
		G, L, final_state = self.QGeometricTensor(input_state, pars, return_all=True)
		Q = np.real(G)
		D = np.imag(G)

		s = self.sloppiness(Q)
		c = self.compatibility(D)
		R = self.asimptoticIncompatibility(Q,D)

		C_SLD, C_H, Delta = self.bounds(Q,D)

		return Q,D,s,c,R,C_SLD,C_H,Delta

	def sloppiness(self,Q):
		return 1/np.linalg.det(Q)

	def compatibility(self,D):
		# return np.linalg.det(D)
		return 2/np.trace( np.dot(np.conj(D).T,D) )

	def asimptoticIncompatibility(self,Q,D):
		return np.abs(np.linalg.eigvals( np.dot(np.linalg.inv(Q),D) ).max())

	def bounds(self,Q,D):
		QI = np.linalg.inv(Q)
		C_SLD = np.trace(QI)
		norm = np.linalg.norm(np.dot(QI,np.dot(D,QI)),ord=1)
		C_H = C_SLD + norm
		Delta = norm/C_SLD
		return C_SLD, C_H, Delta

	def QGeometricTensor(self, input_state, pars, return_all=False):
		# dependency preparation calls
		L = self.SLD(input_state, pars)
		final_state = self.evolve(input_state, pars)

		G = np.zeros((self.npars,self.npars), dtype=np.complex128)
		for i in range(self.npars):
			for j in range(self.npars):
				G[i][j] = np.trace( np.dot(final_state, np.dot(L[i],L[j])) )
		
		if return_all:
			return G,L,final_state
		return G

	def QFIM(self, input_state, pars, return_all=False):
		# dependency preparation calls
		L = self.SLD(input_state, pars)
		final_state = self.evolve(input_state, pars)

		Q = np.zeros((self.npars,self.npars), dtype=np.complex128)
		for i in range(self.npars):
			for j in range(self.npars):
				Q[i][j] = 0.5*np.trace( np.dot(final_state, (np.dot(L[i],L[j])+np.dot(L[j],L[i])) ) )
		
		if return_all:
			return Q,L,final_state
		return Q

	def SLD(self, input_state, pars, epsilon=None):
		if epsilon is None: epsilon=self.DEFAULT_EPSILON
		delta = np.eye(self.npars, dtype=np.complex128)*epsilon
		return np.array([
			2*(self.evolve(input_state,pars+delta[i])-self.evolve(input_state,pars))/epsilon
			for i in range(self.npars)])

	# evolve a state under a given model
	def evolve(self, input_state, pars):
		U = np.array(self(pars),dtype=np.complex128)
		return np.dot(U,np.dot(input_state,np.conj(U).T))

class Sampler:
	def __init__(self, npars, dH):
		"""
		Sampler, that given a certain model, and a certain finite Hilbert space,
		it samples uniformly over the Hilbert space.

		Parameters:
		npars (int): Number of parameters to be estimated.
		dH (int): Dimension of the Hilbert space.
		"""
		self.npars = npars
		self.dH = dH

	def samplePureStates(self,N=1):
		"""Sample a generic state with uniform probability,
		the algorithm samples states in a (0,1)^dH box, then
		only states inside the R=1 dHsphere are kept, and
		renormalized into its surface.

		Args:
			- N: number of states to be sampled

		Returns:
			- numpy array of elements of form (a,b,...,phase_b,phase_c,...)
		"""
		n = 100_000 # instead of doing the whole thing it's done in batches

		# sampling coefficients (a,b,c,d)
		numbers = np.random.uniform(0, 1, (min(10*N,n),self.dH))
		coefficients = numbers[np.sum(numbers**2,axis=1)<1] # selected points withing the sphere
		coefficients = coefficients / np.sqrt(np.sum(coefficients**2,axis=1, keepdims=True)) # normalized
		while len(coefficients)<N:
			numbers = np.random.uniform(0, 1, (min(10*N,n),self.dH))
			numbers = numbers[np.sum(numbers**2,axis=1)<1]
			numbers = numbers / np.sqrt(np.sum(numbers**2,axis=1, keepdims=True))
			coefficients = np.concatenate((coefficients,numbers))

		# sampling phases
		phases = np.random.uniform(0, 2*pi, (N,self.dH-1))

		return np.concatenate((coefficients[:N],phases),axis=1)

	def constructStatisticalOperator(self, states):
		# states of the form (a,b,c,...,phi_b,phi_c,...)
		psi = states[:,:self.dH].astype(np.complex128)
		phases = np.exp(1j * states[:,self.dH:])
		phases = np.hstack((np.ones((states.shape[0], 1)), phases),dtype=np.complex128)
		psi = psi*phases
		return np.conj(psi[:,:,None])*psi[:,None,:] # perform outer product on each state

	def sampleStatisticalOperator(self, N=1):
		states = self.samplePureStates(N)
		return self.constructStatisticalOperator(states)

	def sampleParameters(self, ranges, N=1):
		return np.column_stack( [np.random.uniform(ranges[i][0], ranges[i][1], (N,1)) for i in range(len(ranges))] )


class Plotter:
	"""Class with some plots premade. Block=False makes the given
	plot not stop execution flow.

	Args:
		- s,c,R,Q,D,C_SLD,C_H,Delta (ndarrays): Vectors with respective values.
	"""
	def __init__(self,s=None,c=None,R=None,Q=None,D=None,C_SLD=None,C_H=None,Delta=None):
		self.s = s
		self.c = c
		self.R = R
		self.Q = Q
		self.D = D
		self.C_SLD = C_SLD
		self.C_H = C_H
		self.Delta = Delta

	def plotSC(self,block=True):
		fig = plt.figure()
		plt.title("s-c relation")
		plt.plot(self.s,self.c,'.')
		plt.xlabel("s")
		plt.ylabel("c")
		plt.xscale("log")
		plt.yscale("log")
		# plt.xlim(0,1000)
		# plt.ylim(0,1000)
		plt.show(block=False)

	def plotSR(self,block=True):
		fig = plt.figure()
		plt.title("s-R relation")
		plt.plot(self.s,self.R,'.')
		plt.xlabel("s")
		plt.ylabel("R")
		plt.xscale("log")
		plt.yscale("log")
		# plt.xlim(0,1000)
		# plt.ylim(0,1000)
		plt.show(block=False)

	def plotCR(self,block=True):
		fig = plt.figure()
		plt.title("c-R relation")
		plt.plot(self.c,self.R,'.')
		plt.xlabel("c")
		plt.ylabel("R")
		plt.xscale("log")
		plt.yscale("log")
		# plt.xlim(0,1000)
		# plt.ylim(0,1000)
		plt.show(block=block)

	def plotSCR(self, block=True):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter(self.s, self.c, self.R, marker='.')
		ax.set_xlabel('s')
		ax.set_ylabel('c')
		ax.set_zlabel('R')
		# ax.set_xlim(0,100)
		# ax.set_ylim(0,100)
		ax.set_zlim(0,1)
		plt.title("s-c-R relation")
		plt.show(block=block)

	def plotRHist(self, block=True):
		fig = plt.figure()
		plt.hist(self.R, bins=100, range=[0,1], density=True)
		plt.title("R hist")
		plt.show(block=block)

	def plotBounds(self,mins=True,block=True):
		fig = plt.figure()
		y_SLD = 0
		y_H = 1
		x_min1 = self.C_SLD.min()
		x_min2 = self.C_H.min()
		delta = min(self.Delta[self.C_SLD.argmin()], self.Delta[self.C_H.argmin()])
		r = min(self.R[self.C_SLD.argmin()], self.R[self.C_H.argmin()])
		# print(self.R.min())
		print(f"Difference in R estimation: a: {self.R[self.C_SLD.argmin()]}, b:{self.R[self.C_H.argmin()]}.")
		# print(f"Difference in delta estimation: a: {self.Delta[self.C_SLD.argmin()]}, b:{self.Delta[self.C_H.argmin()]}.")
		if not mins:
			plt.plot(self.C_SLD,[y_SLD]*len(self.C_SLD),'.',c="b")
			plt.plot(self.C_H,[y_H]*len(self.C_H),'.',c="g")
			plt.xscale("log")
			plt.plot([x_min1]*2,[0,1])
			plt.plot([x_min2]*2,[0,1])
		if mins:
			# plt.xlim(0,x_min2+2)
			plt.xlim(x_min1*0.8, 2.2*x_min1)
			plt.ylim(-0.5, 0.5)

			# Plot the two points
			plt.plot(x_min1, 0, '.',c="b", label=r"$C^{SLD}$")  # black circle at x1
			plt.plot(x_min2, 0, '.',c="g", label=r"$C^H$")  # black circle at x2

			# Add <-> arrow (Delta)
			plt.annotate("",
			xy=(x_min2, 0.01), xycoords='data',
			xytext=(x_min1, 0.01), textcoords='data',
			arrowprops=dict(arrowstyle='<->', lw=1.5))
			plt.text((x_min1 + x_min2)/2, 0.1, f"Δ = {str(delta)}", ha='center', va='bottom')

			# Add -> arrow (R)
			plt.annotate("",
			xy=(x_min1*(1+r), -0.01), xycoords='data',
			xytext=(x_min1, -0.01), textcoords='data',
			arrowprops=dict(arrowstyle='->', lw=1.5))
			plt.text((x_min1 + x_min1*(1+self.R.min()))/2, -0.1, f"R = {str(r)}", ha='center', va='top')

		plt.legend()
		plt.title("Precision bounds: C_SLD, C_H")
		plt.show(block=block)

	def plotDeltaHist(self,block=True):
		plt.figure()
		plt.hist(self.Delta, bins=100, range=[0,1], density=True)
		plt.title(r"$\Delta$")
		plt.show(block=block)



