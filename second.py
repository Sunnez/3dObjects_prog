import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from PIL import Image
from PIL import ImageDraw
import scipy.special


points = np.array([[-1, -1, -1],
                   [1, -1, -1],
                   [1, 1, -1],
                   [-1, 1, -1],
                   [-1, -1, 1],
                   [1, -1, 1],
                   [1, 1, 1],
                   [-1, 1, 1]])

P = [[2.06498904e-01, -6.30755443e-07, 1.07477548e-03],
     [1.61535574e-06, 1.18897198e-01, 7.85307721e-06],
     [7.08353661e-02, 4.48415767e-06, 2.05395893e-01]]


def make_bezier(xys):
    n = len(xys)
    combinations = pascal_row(n - 1)

    def bezier(ts):
        result = []
        for t in ts:
            tpowers = (t ** i for i in range(n))
            upowers = reversed([(1 - t) ** i for i in range(n)])
            coefs = [c * a * b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef * p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result

    return bezier


def pascal_row(n):
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result


def var(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    ax.plot(x,y,z)
    ax.scatter3D(x, y, z, alpha=0, edgecolors='r')


def var_poly(i, n, t):

    return scipy.special.comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def var_curse(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([var_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


Z = np.zeros((8, 3))
for i in range(8): Z[i, :] = np.dot(points[i, :], P)
Z = 10.0 * Z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r = [-1, 1]

X, Y = np.meshgrid(r, r)
# plot vertices

# list of sides' polygons of figure
verts = [[Z[0], Z[1], Z[2], Z[3]],
         [Z[4], Z[5], Z[6], Z[7]],
         [Z[0], Z[1], Z[5], Z[4]],
         [Z[2], Z[3], Z[7], Z[6]],
         [Z[1], Z[2], Z[6], Z[5]],
         [Z[4], Z[7], Z[3], Z[0]],
         [Z[2], Z[3], Z[7], Z[6]]]


# plot sides

def init():
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], facecolors=("#%06x" % random.randint(0, 0xFFFFFF)),
                 edgecolors=("#%06x" % random.randint(0, 0xFFFFFF)))
    ax.add_collection3d(
        Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors=("#%06x" % random.randint(0, 0xFFFFFF)),
                         alpha=.25))
    return fig,


mode = [Z[:, 2], Z[:, 1], Z[:, 0]]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


def animate(i):
    ax.view_init(elev=10., azim=i)
    if i % 2 == 0:
        ax.scatter3D(Z[:, 2], Z[:, 1], Z[:, 0], color=("#%06x" % random.randint(0, 0xFFFFFF)))
        ax.plot(Z[:, 2], Z[:, 1], Z[:, 0], color=("#%06x" % random.randint(0, 0xFFFFFF)))
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], facecolors=("#%06x" % random.randint(0, 0xFFFFFF)),
                     edgecolors=("#%06x" % random.randint(0, 0xFFFFFF)))
        ax.add_collection3d(
            Poly3DCollection(verts, facecolors=("#%06x" % random.randint(0, 0xFFFFFF)), linewidths=1,
                             edgecolors=("#%06x" % random.randint(0, 0xFFFFFF)),
                             alpha=.25))
    else:
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], color='w', alpha=0)
        ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], color='w', alpha=0)
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], facecolors='w',
                     edgecolors='w')
        ax.add_collection3d(
            Poly3DCollection(verts, facecolors='white', linewidths=2, edgecolors='w',
                             alpha=0))
    return fig,


im = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
draw = ImageDraw.Draw(im)
ts = [t / 100.0 for t in range(101)]

xys = [(50, 100), (80, 80), (100, 50)]
bezier = make_bezier(xys)
points = bezier(ts)

xys = [(100, 50), (100, 0), (50, 0), (50, 35)]
bezier = make_bezier(xys)
points.extend(bezier(ts))

xys = [(50, 35), (50, 0), (0, 0), (0, 50)]
bezier = make_bezier(xys)
points.extend(bezier(ts))

xys = [(0, 50), (20, 80), (50, 100)]
bezier = make_bezier(xys)
points.extend(bezier(ts))

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

anim.save('second.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
