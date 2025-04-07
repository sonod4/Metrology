from metrologyLib import *
from scipy.linalg import expm
from math import pi

np.random.seed(0)
np.set_printoptions(linewidth=np.inf)


class SU2Model:
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
		self.time = time
		self.representationDimension = representationDimension

		self.J = self.findGenerators(self.representationDimension)

	def __call__(self, pars):
		return expm(-1j * self.time * pars[0] * ( np.cos(pars[1])*self.J[0] + np.sin(pars[1])*self.J[2] ) )

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

		return np.array([J_x,J_y,J_z])



su2 = SU2Model(time=0.3,representationDimension=3)
model = Model(model=su2,npars=2,dH=3)

# ========= Statistical properties =========
STATISTICAL_ANALYSIS = True
if STATISTICAL_ANALYSIS:
	t = time()
	r = model.sampleAnalysis(sampleOverStates=True,sampleOverPars=True,parametersRange=np.array([[0,2*pi],[0,2*pi]]),
								Nstates=100,Npars=10,s=True,c=True,R=True,C_SLD=True,C_H=True,Delta=True)
	print("Calculation time: ", time()-t)
	# print(r["s"].flatten())
	# print(r["c"].flatten())
	# plt.plot(r["s"].flatten(),c="r")
	# plt.yscale("log")
	# plt.plot(r["c"].flatten(),c="g")
	# plt.show()

	PLOT = True
	if PLOT:
		plotter = Plotter(s=r["s"].flatten(),c=r["c"].flatten(),R=r["R"].flatten(),
					C_SLD=r["C_SLD"].flatten(),C_H=r["C_H"].flatten(),Delta=r["Delta"].flatten())

		# plotter.plotSC(r["s"].T,r["c"].T)
		plotter.plotSC(False)
		plotter.plotSR(False)
		plotter.plotCR(False)
		plotter.plotSCR(False)
		plotter.plotRHist(False)
		plotter.plotBounds(mins=True,block=False)
		plotter.plotDeltaHist()





