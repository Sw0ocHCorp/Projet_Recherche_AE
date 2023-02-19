import numpy as np
import matplotlib.pyplot as plt

np.random.seed(562201)
all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
labels = ['x1', 'x2', 'x3']

#MultipleBoxplot
plt.boxplot(all_data, vert=True, patch_artist=True, labels=labels) 
plt.ylabel('observed value')
plt.title('Multiple Box Plot : Vertical Version')
plt.show()

plt.boxplot(all_data, vert=False, patch_artist=True, labels=labels) 
plt.ylabel('observed value')
plt.title('Multiple Box Plot : Horizontal Version')  
