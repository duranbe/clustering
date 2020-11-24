import numpy as np

class KMean:

	def __init__(self,x,k,distance='euclidean'):

		self.x = x     #Input Points
		self.k = k	   #Nb of clusters
		self.dim = x.shape[-1]
		self.n = x.shape[0]
		self.distance = distance
		self.belongs = None
		self.middles = np.random.randint(low=0,high=20,size=(k,self.dim))/2  
		

	def _euclidean_distance(self):

		k = self.k
		n = self.n
		l = self.dim
		middles = self.middles


	    #(k,n,l)
		points = np.broadcast_to(self.x,(k,n,l))

		c = np.zeros((k,n,l))

	    # Need to find a numpy way to do this to avoid loops
	    #(k,n,l)
		for i in range(0,k):
			c[i]=np.broadcast_to(middles[i],(n,l))
	   
	    
	    # Euclidean Distance
	    # (k,n) because distance is a scalar here
		d = np.linalg.norm(points-c,axis=2)
	    
	 
	    # For each point get the argument of the minimum distance
		args=np.argmin(d,axis=0)
	    
		return(args)

	def _new_middles(self):

	    #Calcul of new k using the mean coordinates
	    k = self.k
	    n = self.n
	    l = self.dim

	    middles = self.middles

	    new_middles = np.zeros((k,l))
	    
	    for i in range(0,k):
	        if i in self.belongs:
	            new_middles[i]=np.mean(self.x[self.belongs ==i,:],axis=0)
	        else:
	            new_middles[i]=middles[i]
	            
	    return(new_middles)    

	def run(self,n_iter=5):

		if self.distance == 'euclidean':
			dist_function = self._euclidean_distance

		for i in range(n_iter):
			self.belongs = dist_function()
			self.middles = self._new_middles()
