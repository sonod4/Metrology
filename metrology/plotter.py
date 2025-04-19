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
		"""Class with some plots premade. Block=False makes the given
		plot not stop execution flow.

		Args:
			- s,c,R,Q,D,C_SLD,C_H,Delta (ndarrays): Vectors with respective values.
		"""
		self.data = {}
		if not(data is None):
			self.data = data
		if not(filename is None):
			self.load(filename)

	def load(self, filename, flatten=False):
		with open(filename, 'rb') as f:
			data = pkl.load(f)		
		for k in data.keys():
			if flatten:
				self.data[k] = data[k].flatten()
			else:
				self.data[k] = data[k].T

	# ============================================= Generic relational plots ============================================= #

	def plot2D(self, xlabel, ylabel, xlog=True, ylog=True, diagonal=True, block=True):
		fig = plt.figure()
		plt.title(f"{xlabel}-{ylabel} relation")
		if len(self.data[xlabel].shape)==2:
			NSTATES = 10
			plt.plot(self.data[xlabel][:,:NSTATES],self.data[ylabel][:,:NSTATES],'.',label=f"({xlabel},{ylabel})")
		else:
			plt.plot(self.data[xlabel],self.data[ylabel],'.',label=f"({xlabel},{ylabel})")
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if diagonal:
			xmin = min(self.data[xlabel].min(),self.data[ylabel].min())
			xmax = min(self.data[xlabel].max(),self.data[ylabel].max())
			plt.plot([xmin,xmax],[xmin,xmax],'-r',label="diagonal")
		if xlog:
			plt.xscale("log")
		if ylog:
			plt.yscale("log")
		plt.legend()
		plt.show(block=block)

	def quickPlot2D(self, filename, xlabel, ylabel, xlog=True, ylog=True, diagonal=True, flatten=True, lightweight=False, block=True):
		with open(filename, 'rb') as f:
			data = pkl.load(f)		
		for k in data.keys():
			if flatten:
				if lightweight:
					data[k] = data[k].flatten()[:10_000]
				else:	
					data[k] = data[k].flatten()
			else:
				data[k] = data[k].T

		fig = plt.figure()
		plt.title(f"{xlabel}-{ylabel} relation")
		if len(data[xlabel].shape)==2:
			NSTATES = 10
			plt.plot(data[xlabel][:,:NSTATES],data[ylabel][:,:NSTATES],'.',label=f"({xlabel},{ylabel})")
		else:
			plt.plot(data[xlabel],data[ylabel],'.',label=f"({xlabel},{ylabel})")
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if diagonal:
			xmin = min(data[xlabel].min(),data[ylabel].min())
			xmax = min(data[xlabel].max(),data[ylabel].max())
			plt.plot([xmin,xmax],[xmin,xmax],'-r',label="diagonal")
		if xlog:
			plt.xscale("log")
		if ylog:
			plt.yscale("log")
		plt.legend()
		plt.show(block=block)

	def plot3D(self, xlabel, ylabel, zlabel, block=True):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter(self.data[xlabel], self.data[ylabel], self.data[zlabel], marker='.')
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_zlabel(zlabel)
		# ax.set_xlim(0,100)
		# ax.set_ylim(0,100)
		# ax.set_zlim(0,10)
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_zscale("log")
		plt.title(f"{xlabel}-{ylabel}-{zlabel} relation")
		plt.show(block=block)

	def hist(self, xlabel, bins=100, ranges=None, block=True):
		fig = plt.figure()
		plt.hist(self.data[xlabel], bins=bins, range=ranges, density=True)
		plt.title(f"{xlabel} hist")
		plt.show(block=block)

	# ============================================= Meaningful statistical plots ============================================= #

	def plotBounds(self,Nstates=1,Npars=1,metrics=["R","T"],sort_by=None,log=False,block=True):
		"""
		Plots up to min(Nstates, Npars) subplots, each showing the chosen metrics
		as functions of the remaining (unfixed) axis.

		• If Nstates < Npars, we produce one subplot per state (i.e. fixing each row).
		• Otherwise, one subplot per parameter (i.e. fixing each column).
		• metrics: list of keys in self.data to plot.
		"""
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

			if fix_states:
				# Fix a state → vary parameters on x-axis
				state = idx
				x = list(range( min(Npars,self.data[metrics[0]].shape[1]) ))  # or custom parameter labels
				ax.set_xlabel("Parameter index")
				ax.set_title(f"State {state}")
				if log:
					ax.set_yscale("log")
				if not(sort_by is None):
					order = np.argsort(self.data[sort_by][state,:Npars])
				for metric in metrics:
					if not(sort_by is None):
						y = self.data[metric][state,:Npars][order]
					else:
						y = self.data[metric][state,:Npars]
					ax.plot(x, y, marker='.', label=metric)
			else:
				# Fix a parameter → vary states on x-axis
				param = idx
				x = list(range( min(Nstates,self.data[metrics[0]].shape[0]) ))  # or custom state labels
				ax.set_xlabel("State index")
				ax.set_title(f"Parameter {param}")
				if log:
					ax.set_yscale("log")
				if not(sort_by is None):
					order = np.argsort(self.data[sort_by][:Nstates,param])
				for metric in metrics:
					# take column `param` from each row
					if not(sort_by is None):
						col = self.data[metric][:Nstates,param][order]
					else:
						col = self.data[metric][:Nstates,param]
					ax.plot(x, col, marker='.', label=metric)

			ax.legend()
			ax.grid(True)

		# Remove any unused subplots
		for j in range(nplots, len(axes_flat)):
			fig.delaxes(axes_flat[j])

		plt.tight_layout()
		plt.show(block=block)