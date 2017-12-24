__author__ = 'Erdem'

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

for kk in xrange(1,10000):
    image = PIL.Image.open('x.ps')
    image.thumbnail((160, 160))
    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = image[:, :, :3]
    print(kk)
# plt.imshow(image)
# plt.show()