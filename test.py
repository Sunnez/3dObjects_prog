from PIL import Image
from PIL import ImageDraw
from matplotlib import animation
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
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
    draw.polygon(points, fill='red')
    im.save('out.png')
