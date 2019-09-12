from PIL import Image
import numpy as np
import pandas as  pd 
import matplotlib.pyplot as plt



gambar=Image.open('baby.jpg').convert('L') #RGBA CMYK
gambar=np.array(gambar)
# print(gambar.shape)

# plt.imshow(gambar,cmap='gray')
# plt.show()

out=Image.fromarray(gambar,'L')
out.show()

