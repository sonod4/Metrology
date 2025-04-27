import numpy as np
from numpy import e,pi,sqrt
import matplotlib.pyplot as plt
import pickle as pkl
import math


# ================================================================================================= #
# ============================================ PLOTTER ============================================ #
# ================================================================================================= #

class Plotter:
	def __init__(self,data=None,filename=None):
		"""
		Class with some plots premade. Block=False makes the given
		plot not stop execution flow.

		Parameters:
		data (dict): Dictionary with the metrics given by Analyzer.
		filename (string): Name of the file in which the dictionary has been pickled into.
		"""
		self.data = {}
		if not(data is None):
			self.data = data
		if not(filename is None):
			self.data = self.load(filename)

	def load(self, filename):
		with open(filename, 'rb') as f:
			data = pkl.load(f)
		return data

	# ============================================= Generic relational plots ============================================= #

	def formatData(self,filename,xlabel,ws=None,states=None,pars=None):
		data = self.data
		if not(filename is None):
			data = self.load(filename)

		NUMBER_OF_WEIGHTS = data[xlabel].shape[0]
		NUMBER_OF_STATES = data[xlabel].shape[1]
		NUMBER_OF_PARS = data[xlabel].shape[2]

		if ws is None:
			ws = np.arange(NUMBER_OF_WEIGHTS)
		if states is None:
			states = np.arange(NUMBER_OF_STATES)
		if pars is None:
			pars = np.arange(NUMBER_OF_PARS)
		states = np.array(states)
		pars = np.array(pars)
		ws = np.array(ws)
		return data, states, pars, ws

	def plot2D(self, xlabel, ylabel, filename=None, ws=None, states=None, pars=None,
				xlog=False, ylog=False, diagonal=True, seeStates=False, block=True):
		data, states, pars, ws = self.formatData(filename,xlabel,ws,states,pars)

		x = data[xlabel][np.ix_(ws,states,pars)]
		y = data[ylabel][np.ix_(ws,states,pars)]
		
		if seeStates:
			x = x.transpose(1,0,2).reshape(states.shape[0],ws.shape[0]*pars.shape[0])
			y = y.transpose(1,0,2).reshape(states.shape[0],ws.shape[0]*pars.shape[0])
		else:
			x = x.flatten()
			y = y.flatten()

		fig = plt.figure()
		plt.title(f"{xlabel}-{ylabel} relation")
		plt.plot(x,y,'.',label=f"({xlabel},{ylabel})")

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if diagonal:
			xmin = min(x.min(),y.min())
			xmax = min(x.max(),y.max())
			plt.plot([xmin,xmax],[xmin,xmax],'-r',label="diagonal")
		if xlog:
			plt.xscale("log")
		if ylog:
			plt.yscale("log")
		plt.legend()
		plt.show(block=block)

	def plot3D(self, xlabel, ylabel, zlabel, filename=None, ws=None, states=None, pars=None, seeStates=False, block=True):
		data, states, pars, ws = self.formatData(filename,xlabel,ws,states,pars)
		x = data[xlabel][np.ix_(ws,states,pars)]
		y = data[ylabel][np.ix_(ws,states,pars)]
		z = data[zlabel][np.ix_(ws,states,pars)]
		
		if seeStates:
			x = x.transpose(1,0,2).reshape(states.shape[0],ws.shape[0]*pars.shape[0])
			y = y.transpose(1,0,2).reshape(states.shape[0],ws.shape[0]*pars.shape[0])
			z = z.transpose(1,0,2).reshape(states.shape[0],ws.shape[0]*pars.shape[0])
		else:
			x = x.flatten()
			y = y.flatten()
			z = z.flatten()

		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter(x, y, z, marker='.')
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_zlabel(zlabel)
		# ax.set_xscale("log")
		# ax.set_yscale("log")
		# ax.set_zscale("log")
		plt.title(f"{xlabel}-{ylabel}-{zlabel} relation")
		plt.show(block=block)

	def hist(self, xlabel, bins=100, ranges=None, filename=None, ws=None, states=None, pars=None, block=True):
		data, states, pars, ws = self.formatData(filename,xlabel,ws,states,pars)

		x = data[xlabel][np.ix_(ws,states,pars)].flatten()

		fig = plt.figure()
		plt.hist(x, bins=bins, range=ranges, density=True)
		plt.title(f"{xlabel} hist")
		plt.show(block=block)

	# ============================================= Meaningful statistical plots ============================================= #

	def plotBounds(self,metrics=["R","T"],sort_by=None,ws=None,states=None,pars=None,log=False,filename=None,block=True):
		"""
		Plots up to min(Nstates, Npars) subplots, each showing the chosen metrics
		as functions of the remaining (unfixed) axis.

		If Nstates < Npars, it produces one subplot per state (i.e. fixing each row).
		Otherwise, one subplot per parameter (i.e. fixing each column).
		metrics: list of keys in self.data to plot.
		"""
		data, states, pars, ws = self.formatData(filename,metrics[0],ws,states,pars)
		
		Nstates = states.shape[0]
		Npars = pars.shape[0]

		# Determine how many subplots
		nplots = min(Nstates, Npars)

		# Compute grid size (as square as possible)
		nrows = int(math.floor(math.sqrt(nplots)))
		ncols = int(math.ceil(nplots / nrows))
		# Guarantee enough rows
		if nrows * ncols < nplots:
			nrows += 1

		fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
		axes_flat = axes.flatten()

		# Decide which axis is “fixed”
		fix_states = (Nstates < Npars)

		for idx in range(nplots):
			ax = axes_flat[idx]

			# fixing states or params
			if fix_states:
				# Fix a state → vary parameters on x-axis
				ax.set_xlabel("Parameter index")
				ax.set_title(f"State {idx}")
				args = np.ix_(ws,[idx],pars)
			else:
				# Fix a parameter → vary states on x-axis
				ax.set_xlabel("State index")
				ax.set_title(f"Parameter {idx}")
				args = np.ix_(ws,states,[idx])
			if log:
				ax.set_yscale("log")

			# data
			if not(sort_by is None):
				order = np.argsort(data[sort_by][args].flatten())
			for metric in metrics:
				if not(sort_by is None):
					y = data[metric][args].flatten()[order]
				else:
					y = data[metric][args].flatten()
				# ax.plot(x, y, marker='.', label=metric)
				ax.scatter(range(len(y)),y, s=2, label=metric)

			# ax.set_ylim(0,5)

			ax.legend()
			ax.grid(True)

		# Remove any unused subplots
		for j in range(nplots, len(axes_flat)):
			fig.delaxes(axes_flat[j])

		if not(filename is None):
			plt.suptitle(filename)
		plt.tight_layout()
		plt.show(block=block)

	def plotW2D(self, metrics=["R","T"],metrics2=[], states=None, pars=None, xlog=False, y2log=False, filename=None, block=True):
		data, states, pars, ws = self.formatData(filename,metrics[0],None,states,pars)
		
		Nstates = states.shape[0]
		Npars = pars.shape[0]

		# Determine how many subplots
		nplots = Nstates*Npars

		# Compute grid size (as square as possible)
		nrows = Nstates
		ncols = Npars
		# Guarantee enough rows
		if nrows * ncols < nplots:
			nrows += 1

		fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
		axes_flat = axes.flatten()

		x = data["W"][:,-1,-1]
		for s in states:
			for p in pars:
				ax = axes_flat[s*Npars+p]

				# Fix a state → vary parameters on x-axis
				ax.set_xlabel("w")
				ax.set_title(f"State {s} | pars {p}")
				if xlog:
					ax.set_xscale("log")
				cs = ["C0","C1","C4","C5","C6"]
				for m in range(len(metrics)):
					y = data[metrics[m]][:,s,p]
					ax.scatter(x, y, s=10, label=metrics[m], c=cs[m%5])

				# optional second metrics
				if len(metrics2)>0:
					cs = []
					ax2 = ax.twinx()
					if y2log:
						ax2.set_yscale("log")
					cs = ["C2","C3","C7","C8","C9"]
					for m in range(len(metrics2)):
						y = data[metrics2[m]][:,s,p]
						ax2.scatter(x, y, s=10, label=metrics2[m], c=cs[m%5])
				# ax.set_ylim(0,5)

				ax.legend()
				if len(metrics2)>0:
					ax2.legend()
				ax.grid(True)

		# Remove any unused subplots
		for j in range(nplots, len(axes_flat)):
			fig.delaxes(axes_flat[j])

		if not(filename is None):
			plt.suptitle(filename)
		plt.tight_layout()
		plt.show(block=block)





