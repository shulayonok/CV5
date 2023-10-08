import numpy as np
from PIL import Image
import funcs

# Считываем изображение
image = np.array(Image.open("geometric.jpg"))
funcs.apply(image)