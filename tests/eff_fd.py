import cv2
import os
import sys
sys.path.append('.')

from src.eff_fd import EffFD

def test(img_name, theoretical_dim):
    img_path = os.path.join('tests', 'img', img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.bitwise_not(img)

    dim = EffFD.getHausdorffDimension(img)
    
    print(f'\n## {img_name} ##')
    print(f'. Theoretical Hausdorff Dimension: {theoretical_dim}')
    print(f'. Approximated Hausdorff Dimension: {dim}')
    print(f'. Error: {100*abs(theoretical_dim-dim)/theoretical_dim:.2f}%')

test('sierpinski.png',1.5850)
test('sierpinski2.png',1.5850)
test('koch.png',1.2619)
test('koch2.png',1.2619)
test('2d.png',2.0)