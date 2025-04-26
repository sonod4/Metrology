
# Metrology library
This python library is aimed to be used as a tool to provide generic analysis on **pure** models.

**Index:**
- [Library Classes](#library-classes)
	- [Analyzer](#analyzer)
	- [Sampler](#sampler)
	- [Plotter](#plotter)
- [Metrics](#metrics)
- [Model](#model)
- [Usage Examples](#usage-examples)

## Library Classes
The library is made of 3 classes that can be used.
- [Analyzer](#model)
- [Sampler](#sampler)
- [Plotter](#plotter)
### Analyzer
This class is the main tool that do the analysis. The class definition is
```def __init__(self,model,suppress)```
Where it has to receive the following parameters:
- **model**(Model type class): Is a callable object, that takes in input a vector of the parameters to be estimated, and returns the unitary evolution valued at such parameters+(some additional hidden parameters that may be hidden in the object, see [Model](#model) for further details).
- **suppress**(ndarray, shape:(2)): List of parameters in the nbit to suppress.

Given the parametrization

$$  \begin{bmatrix}
    a \\
    b \\
    c \\
    \end{bmatrix} $$
    
the first number specify how many parameters(a,b,...) from the bottom to set to 0, the second number specify how many *additional* phases to set to 0. So for example [1,2] reduces

$$  \begin{bmatrix}
    a \\
    b e^{i\phi_b} \\
    c e^{i\phi_c} \\
    d e^{i\phi_d} \\
    f e^{i\phi_f} \\
    \end{bmatrix} 
\mapsto
 \begin{bmatrix}
    a \\
    b e^{i\phi_b} \\
    c e^{0} \\
    d e^{0} \\
    0 \\
    \end{bmatrix}$$
    
The main method to be called is 
```python
def fastSampleAnalysis(self, state=None, pars=None, parametersRange=None,
			Nstates=1,Npars=1, metrics=[], Ws=None, epsilon=None,
			save=None, parallelize=True, verbose=True)
```
This method returns a dictionary(a map key-values) with the asked statistical informations **calculated for each sample combination**(so a matrix of numbers).
The function inputs are:
- **state**(ndarray): If we want to work with a specific state($\rho$), or a list of states, we can pass them here. Shape of input is therefore either (dH,dH) or (#states,dH,dH).
- **pars**(ndarray): If we want to work with a specific parameters choice, or a list of parameters, we can pass them here. Shape of input is therefore either (npars) or (#parametersWanted,npars).
- **parametersRange**(ndarray): Vector of ranges, over which the respectively parameters are sampled over. (i.e. ```np.array([ [0,0.01],[0,2*pi] ])``` ). It can be useful if we want to sample small values for some(or all) the parameters. If pars is not passed this argoument is mandatory.
- **Nstates**(int): It specify the number of states to randomly(uniformly) sample over Hilbert space $H$(or $H\otimes K$ in case the model uses entangled probes).
- **Npars**(int): It specify the number of parameter choices to randomly(uniformly) sample over the ranges defined in *parametersRange*.
- **metrics**(list of strings): Name of metrics to calculate for each sample. Supported metrics are ["s","c","R","T","C_SLD","C_H","C_W","Delta_R_T"](refer to [Metrics](#metrics) for their definitions).
- **Ws**(ndarray): Must be real positive matrix(or vector of matrices) of shape either (npars,npars) or (#ws,npars,npars). Default value is the identity.
- **epsilon**(float): In calculating derivatives, it's used an incremental step, *epsilon* is the size of the step which can be adjusted. (Default value is 0.00000001.)
- **save**(string): If passed saves the result dictionary into a file with that name(pickle is used).
- **parallelize**(bool): The calculation for the SLDs are parallelized by default with Thread. If set to False this is done sequentially instead.
- **verbose**(bool): Default value is True, and prints time elapsed in each major step. If False the print are suppressed.

Returns:
- A dictionary with the metrics asked for, in the call of the function.

**Note**: The returned object is a dictionary of *matrices*, where the rows are the different states, while the columns are the different parameters choice. Therefore for example ```r["s"]``` will have shape (Nstates,Npars).
For plotting reasons(wanting to ignore specific states, or parameters choices), it can be useful to ```.flatten()``` the matrix.

This is the main function that is expected to be called. Other than this there are also some functions which may be of interest(but mostly can be ignored).

### Sampler
This method is an internal library object, and it's not supposed to be used(but it is possible to). The class definition is
```python
def __init__(self, model, suppress):
```
This works the same as [Analyzer](#analyzer)

Useful methods which could be called are the following.
```python
def sampleStatisticalOperators(self, N=1):
```
Input: Number of states to sample.

Returns: Vector of statistical operators of shape (N, dH, dH).

```python
def sampleEntangledStatisticalOperators(self, N=1):
```
Input: Number of states to sample.

Returns: Vector of statistical operators of shape (N, dH\*dK, dH\*dK), made as

$$\rho_i \equiv (\sum\limits_{k=1}^{n} \ket{\psi_k}\ket{\phi_k}) (\sum\limits_{k=1}^{n} \bra{\psi_k}\bra{\phi_k})$$

with n given by the *entanglement_n* parameter to be specified in the Model(see [Model](#model) for further details).

### Plotter
This object is a container of interesting prebuilt plots. The class definition is
```python
def __init__(self,data=None,filename=None):
```
Inputs:
- **data**(dictionary): Dictionary with as keys the metrics(the one returned by Analyzer).
- **filename**(string): Name of the pickle file into which the data dictionary has been pickled.

The class can be used either with a specific dictionary, which can be loaded either in the constructor, or on the fly while calling plot functions.

Here a list of the prebuilt plot functions.
```python
def plot2D(self, xlabel, ylabel, filename=None, ws=None, states=None, pars=None,
				xlog=False, ylog=False, diagonal=True, seeStates=False, block=True):
```
Inputs:
- **xlabel**(string): Name of the metric to put on the x.
- **ylabel**(string): Name of the metric to put on the y.
- **filename**(string): If given, loads and plots the data in the file specified.
**Note**: this does not override the data currently stored(loaded in the constructor) in the class.
- **ws**(ndarray): Vector specifying which(by index) weights select to display.
- **states**(ndarray): Vector specifying which(by index) states select to display.
- **pars**(ndarray): Vector specifying which(by index) pars select to display.
- **xlog,ylog**(bool): Sets the relative axis with a logarithmic scale.
- **diagonal**(bool): Display a red line as the main diagonal(for reference).
- **seeStates**(bool): Displays different probe states with different colors.
**Note**: if many states are selected, the plot gets slow and messy.
- **block**(bool): If False multiple plots can be shown at the same time. If True each plot will block execution(as is the matplotlib default behaviour).


```python
def plot3D(self, xlabel, ylabel, zlabel, filename=None, ws=None, states=None, pars=None,
			seeStates=False, block=True):
```
Inputs: same as plot2D(zlabel is just the metric to put on z axis).

```python
def hist(self, xlabel, bins=100, ranges=None, filename=None,
					ws=None, states=None, pars=None, block=True):
```
Inputs: most are the same as plot2D. ```bins``` is the number of bins, and ```range``` specify the range of the histogram.

```python
def plotBounds(self,metrics=["R","T"],sort_by=None,ws=None,states=None,pars=None,
				log=False,filename=None,block=True):
```
This function displays a grid of plots. Each plot has a fixed state/params. This depends on how much states and params are asked to be considered. The smaller of the two *is* the number of plots, and each plot has that one fixed, and plots the metrics for every possible value of the other.
Example: number of states is 5 and number of params is 1000. A window with 5 plots is generated. Each plot has 1000*(number of ws) points, one for each parameter(and weight) combination.

Inputs(the ones different from plot2D):
- **metrics**(list of strings): The metrics to plot with fixed state/parameter for all the parameters/states.
- **sort_by**(string): Specify a metric for which the various points in the plot are sorted by.
**Note**: the metric doesn't have to be one of the plotted metrics. For example ```plotBounds(metrics=["R","T"], sort_by="C_SLD", ws=[5,6],states=range(5), pars=range(1000))``` will plot R and T values, but sort them in ascending order with respect to the "optimality"($C_{\text{SLD}}$ value) of the state for the given parameter. This can be interesting to see for example the incompatibility change between optimal to worse probes. (Also, here there will be 5 plots, one for each of the first 5 states, each with 2000 points, corresponding to the firsts 1000 params, for the weight matrices at index 5 and 6).
- **log**(bool): If True the y is set to logarithmic scale.


```python
def plotW2D(self, metrics=["R","T"],metrics2=[], states=None, pars=None,
				xlog=False, y2log=False, filename=None, block=True):
```
This function displays a window with a grid of plots corresponding to the cartesian product of the selected states and params. For each combination a plot with the asked metrics is plotted. The x axis is the weight. So the graphs display the dependency of the various metrics, with respect to W, valued at different state/params choices.

Inputs(the ones different from plot2D):
- **metrics**(list of strings): The metrics to plot with fixed state/parameter for all the W.
- **metrics2**(list of strings): The metrics to plot with fixed state/parameter for all the W, *on a secondary yaxis*.
- **xlog**(bool): Set x axis to log scale.
- **y2log**(bool): Set y axis, for the secondary metrics, to a log scale.


## Metrics
The metric supported are ["s","c","R","T","C_SLD","C_W","Delta_R_T"].

They are calculated as:
- **sloppiness**: $s = \frac{1}{\text{Det} Q}$
- **compatibility**: $c = \frac{2}{\text{Tr}[D^\dagger D]}$
- **asymptotic incompatibility**: $R = || iQ^{-1}D ||_\infty$
- **T**: $T = || \sqrt{W}Q^{-1}DQ^{-1}\sqrt{W} ||_1$
- **C_SLD**: $C_{\text{SLD}} = \text{Tr}[WQ^{-1}]$
- **C_W**: $C_{\text{W}} =  C_{\text{SLD}} + T$
- **Delta_R_T**: $\Delta_{R-T} \equiv R-T$

## Model
### General structure
The model is the key object, needed for making the library work. Since it's generic, it has to be user defined, with some restrictions. The base Model should look like.
```python
class MyModel:
	def __init__(self,someFixedParameters):
		# mandatory class variables
		self.dH = 2 # dim of Hilbert space
		self.npars = 2 # number of pars to estimate
		# optional class variables
		self.dK = 3 # dim of auxiliary space
		self.entanglement_n = 10 # number of entangled pairs

		# extra parameters known(fixed) in the model
		self.someFixedParameters = someFixedParameters # i.e. time
		# some generators
		self.J = ...
		
	def __call__(self, pars):
		return torch.matrix_exp(-1j * self.time * pars[:,0][:,None,None] * \
		( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*self.J[2] ) )
```
### Mandatory elements
The class **must have** the variables ```self.dH``` and ```self.npars```, which specify respectively the dimension of the Hilbert space on which the model is acting(the dimension of the probes), and the number of parameters to be estimated in the model.
Additionally it **has to be implemented** the ```__call__``` method. This method **has to support broadcasting**, meaning its taking as input a torch tensor of shape (Npars,npars) (Npars is how many parameters choices are sampled, npars is the number of parameters the model has to estimate i.e. 2 or 3), and **has to output a torch tensor of shape (Npars,dH,dH)**(or (Npars,dH\*dK,dH\*dK) if the auxiliary space is being used). 
The function matrix_exp of torch is advised.

### Use of auxiliary spaces
If the model work on more than 1 system(i.e. 2 systems), it can be implemented in the following way.
1) Specify the ```self.dK``` and ```self.entanglement_n``` variables.
**Note**: These two variables can be in general not specified. If they are specified, and ```self.dK```$\neq 0$, then the code automatically sample over both spaces.
2) The ```__call__``` function has to output some kind of unitary **on the tensor space** $H\otimes K$. For example interesting cases are $U' \equiv U\otimes \mathbb{I}$, or $U"\equiv U \otimes U$.

These two cases are implementable as follows:
Case $U' \equiv U\otimes \mathbb{I}$:
```python
def __call__(self, pars):
	U = matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*self.J[2] ) )
	I_k = tensor(np.eye(self.dK))
	return kron(U, I_k)
```
Case $U" \equiv U\otimes U$:
```python
def __call__(self, pars):
	U = matrix_exp(-1j * self.time * pars[:,0][:,None,None] * ( cos(pars[:,1])[:,None,None]*self.J[0] + sin(pars[:,1])[:,None,None]*self.J[2] ) )
	kron_intermediate = torch.einsum('nij,nkl->nikjl', U, U)
	result = kron_intermediate.reshape(U.shape[0], U.shape[1] * U.shape[1], U.shape[2] * U.shape[2])
	return result
```
### Generators for SU(2) in generic representation
In the code we may want to make our class "dimension generic". For example we could have
```python
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
		self.dH = representationDimension

		self.J = self.findGenerators(self.dH)
```
Where the ```self.findGenerators(self.representationDimension)``` function returns a vector with the 3 generators of SU(2), in a generic representation. The implementation is as follows:
```python
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

	return tensor([J_x,J_y,J_z])
```

## Usage Examples
In the main folder there are a bunch of working scripts as examples. The ones that ends with "*Weighted*" samples models over different weight matrices, the others are just base implementations of the various models.

The idea is to always:
1) Define the [Model](#model)
2) Create an [Analyzer](#analyzer), and run a personalized ```fastSampleAnalysis```(storing the data in a folder).
3) Plot the data using the [Plotter](#plotter)

**Note**: When running the first time on specific models, the library compiles a function, so it may take 30 to 60 seconds more than usual. The compilation will be cached, so successive executions will be quicker.

## Examples of Plots
There are no files in the main that shows examples of plots, so here are a few example plots.
```python
pl = Plotter(filename="data/W2scrambler_100_50_100.pkl")
pl.plot2D("R","T",ws=range(2),states=range(3),pars=range(2),diagonal=False)
pl.hist("T",ws=range(100))
pl.plotBounds(sort_by="C_SLD",ws=[50],states=range(10),pars=range(100))
pl.plotW2D(metrics=["R","T"],metrics2=["C_SLD"],states=range(3),pars=range(3),xlog=True,y2log=True)

```
