from metrologyLib import *
from scipy.linalg import expm
from math import pi

np.random.seed(0)
np.set_printoptions(linewidth=np.inf)

class ScramblingModel:
	def __init__(self,n,gamma):
		"""
		Model implementing the scrambling model. Parameters are in the form
		of (lambda_1,lambda_2)

		Parameters:
		n (np.ndarray): Direction of the scrambling hamiltonian.
		gamma (float): Intensity of the scrambling hamiltonian.
		"""
		self.n = n
		self.gamma = gamma

		# Pauli Matrices
		self.PM = np.array([ [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]   ], dtype=np.complex128)

	def __call__(self, pars):
		# PM[2] is sigma_z
		return np.dot( expm(-1j*pars[1]*self.PM[2]),
				np.dot(expm(-1j * self.gamma * np.sum(self.n.reshape(len(self.n),1,1)*self.PM,axis=0) ),
				expm(-1j*pars[0]*self.PM[2])) )



# ========= Model creation =========
pars = np.array([1,3])
n = np.array([1/sqrt(2),1/sqrt(2),0])
gamma = 0.6
scrambler = ScramblingModel(n,gamma)

# ========= Statistical analyzers =========
model = Model(model=scrambler,npars=2,dH=2)

# ========= Analysis on a state =========
STATE_ANALYSIS = False
if STATE_ANALYSIS:
	sampler = Sampler(npars=2,dH=2)
	psi = np.array([ [1/sqrt(2),1/sqrt(2),3.141/2] ]) # states of the form (a,b,c,...,phi_b,phi_c,...)
	rho = sampler.constructStatisticalOperator(psi)[0]
	print("rho: ", rho)

	print("Analysis:")
	Q,D,s,c,R = model.analyzeState(input_state=rho,pars=pars)
	print(Q)
	print(D)
	print(s,c,R)
	print("Analysis 2(equivalent):")
	# r = model.sampleAnalysis(sampleOverStates=False,sampleOverPars=False,parametersRange=np.array([[0,2*pi],[0,2*pi]]),s=True,c=True,R=True)
	r = model.sampleAnalysis(sampleOverStates=False,sampleOverPars=False,state=rho,pars=pars,s=True,c=True,R=True)
	print(r)

# ========= Statistical properties =========
STATISTICAL_ANALYSIS = True
if STATISTICAL_ANALYSIS:
	t = time()
	# r = model.sampleAnalysis(sampleOverStates=True,sampleOverPars=False,pars=[0.1,2.3],Nstates=10,s=True,c=True,R=True)
	r = model.sampleAnalysis(sampleOverStates=True,sampleOverPars=True,parametersRange=np.array([[0,2*pi],[0,2*pi]]),Nstates=20,Npars=100,s=True,c=True,R=True)
	print("Calculation time: ", time()-t)
	# print(r)
	print(r["s"].flatten())
	print(r["c"].flatten())

	PLOT = True
	if PLOT:
		plotter = Plotter(s=r["s"].flatten(),c=r["c"].flatten(),R=r["R"].flatten())

		plotter.plotSC(False)
		plotter.plotSCR(False)
		plotter.plotRHist()







