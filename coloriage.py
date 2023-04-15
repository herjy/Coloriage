import sys
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


img = plt.imread(sys.argv[1])

k1 = np.array([[[1,0,-1],[0,0,0],[-1,0,1]], [[1,0,-1],[0,0,0],[-1,0,1]], [[1,0,-1],[0,0,0],[-1,0,1]]])/12
k2 = np.array([[[0,1,0],[1,0,-1],[0,-1,0]], [[0,1,0],[1,0,-1],[0,-1,0]], [[0,1,0],[1,0,-1],[0,-1,0]]])/12



img = nd.convolve(img,k1)
img = nd.convolve(img,k2)


plt.imshow(img)
plt.show()

def colour_bin(img, N_bin):
	''' Decrease the tone richness of an image to N_bins

	Parameter
	---------
	img: numpy ndarray
		colour image (shape n1xn2x3)
	N_bin: int
		number of bins for tone resolution

	Returns
	-------
	sub_img: numpy ndarray
		a version of img with the colour resolution tuned down
	'''

	sub = img * (N_bin-1)/np.max(img)
	sub_img = np.floor(sub)
	sub_img *= 255/np.max(sub_img)
	return sub_img.astype(int)



sub = colour_bin(np.array(img, dtype = float), 3)


plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.imshow(sub)
plt.axis('off')
plt.subplot(133)
plt.imshow(img-sub)
plt.show()