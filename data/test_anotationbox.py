import matplotlib.pyplot as plt 
from matplotlib.offsetbox import TextArea, AnnotationBbox

fig, ax = plt.subplots()

offsetbox = TextArea("Test 1")

xy = (10,5)

plt.plot(xy[0], xy[1])
ab = AnnotationBbox(offsetbox, xy,
                    xybox=(-20, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

plt.show()
