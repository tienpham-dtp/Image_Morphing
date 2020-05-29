'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''


import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from PIL import Image
import imageio

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # Tips: use Delaunay() function to get Delaunay triangulation;
  # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
    frame_num = warp_frac.shape[0] 
    im1_morphed = np.zeros((frame_num, im1.shape[0], im1.shape[1], im1.shape[2]))
    im2_morphed = im1_morphed.copy()
    morphed_im = im1_morphed.copy()
    
    for f in range(frame_num):
        intermediate_im = (1-warp_frac[f]) * im1_pts + warp_frac[f] * im2_pts
        delaunay_obj = spatial.Delaunay(intermediate_im)
        intermediate_simplex = delaunay_obj.simplices
        
        for y in range(im1.shape[0]):
            for x in range(im1.shape[1]):
                simplex_idx = delaunay_obj.find_simplex(np.array([x,y]))
                
                matrix = np.transpose(intermediate_im[intermediate_simplex[simplex_idx]])
                matrix = np.vstack((matrix,np.ones((1,3))))
                barycentric_coor = np.dot(np.linalg.inv(matrix), np.array([[x],[y],[1]]))
                
                matrix_im1 = np.transpose(im1_pts[intermediate_simplex[simplex_idx]])
                matrix_im1 = np.vstack((matrix_im1,np.ones((1,3))))

                matrix_im2 = np.transpose(im2_pts[intermediate_simplex[simplex_idx]])
                matrix_im2 = np.vstack((matrix_im2,np.ones((1,3))))
                
                corr1_coor = np.dot(matrix_im1, barycentric_coor)
                corr2_coor = np.dot(matrix_im2, barycentric_coor)
                
                y1 = int(corr1_coor[0])
                x1 = int(corr1_coor[1])
                
                y2 = int(corr2_coor[0])
                x2 = int(corr2_coor[1])

                
                im1_morphed[f, y, x, 0] = im1[int(x1), int(y1), 0]
                im1_morphed[f, y, x, 1] = im1[int(x1), int(y1), 1]
                im1_morphed[f, y, x, 2] = im1[int(x1), int(y1), 2]
                
                im2_morphed[f, y, x, 0] = im2[int(x2), int(y2), 0]
                im2_morphed[f, y, x, 1] = im2[int(x2), int(y2), 1]
                im2_morphed[f, y, x, 2] = im2[int(x2), int(y2), 2]
                
        morphed_im[f, :, :, :] = (1 - dissolve_frac[f]) * im1_morphed[f, :, :, :] + dissolve_frac[f] * im2_morphed[f, :, :, :]


    # discretization
    np.clip(morphed_im, 0, 255, out = morphed_im)

    
    return morphed_im.astype('uint8')

lst = [[0,0],[0,0],[150,0], [150,0], [0,150], [0,150], [300,0], [300,0], [0,300], [0,300], [300,300],[300,300], [150,300],[150,300],[300,150],[300,150], [80.66935483870968, 179.57258064516128], [101.2338709677419, 150.29838709677415], [115.26612903225806, 178.24193548387098], [139.82258064516128, 162.27419354838707], [171.1532258064516, 182.2338709677419], [186.39516129032262, 160.94354838709677], [213.73387096774195, 182.2338709677419], [224.9838709677419, 146.30645161290317], [97.96774193548386, 184.89516129032256], [111.87903225806457, 166.26612903225805], [191.11290322580646, 187.55645161290323], [207.68548387096774, 164.9354838709677], [95.30645161290322, 168.92741935483872], [115.87096774193543, 140.9838709677419], [188.4516129032258, 174.25], [203.69354838709677, 140.9838709677419], [115.26612903225806, 216.8306451612903], [150.46774193548384, 203.52419354838707], [161.83870967741933, 218.1612903225806], [178.41129032258067, 202.1935483870967], [136.55645161290323, 222.1532258064516], [163.77419354838713, 218.16129032258058], [112.6048387096774, 247.43548387096772], [145.14516129032262, 232.79838709677415], [157.8467741935484, 252.75806451612902], [178.41129032258067, 230.13709677419348], [131.2338709677419, 256.75], [165.10483870967738, 234.12903225806446], [136.55645161290323, 239.4516129032258], [162.44354838709677, 222.15322580645156], [131.2338709677419, 278.0403225806451], [162.44354838709677, 248.766129032258], [79.33870967741935, 263.4032258064516], [106.55645161290323, 238.12096774193543], [59.3790322580645, 251.4274193548387], [73.29032258064518, 219.49193548387092], [46.07258064516127, 227.4758064516129], [57.32258064516128, 166.26612903225805], [14.137096774193537, 196.8709677419355], [46.67741935483872, 83.76612903225805], [35.427419354838705, 65.13709677419354], [48.00806451612908, 3.927419354838719], [149.86290322580646, 31.870967741935488], [153.12903225806446, 62.475806451612925], [236.35483870967744, 55.82258064516128], [260.91129032258056, 11.911290322580669], [278.93548387096774, 114.37096774193549], [246.27419354838702, 111.70967741935482], [273.61290322580646, 160.94354838709677], [256.9193548387097, 144.97580645161287], [282.9274193548387, 191.54838709677415], [254.25806451612897, 174.24999999999994], [253.6532258064516, 227.4758064516129], [247.60483870967738, 196.87096774193543], [217.72580645161293, 254.08870967741933], [224.9838709677419, 211.508064516129], [183.1290322580645, 267.39516129032256], [206.35483870967738, 234.12903225806446]]
im1_pts = [lst[i] for i in range(len(lst)) if i%2 == 0]
im2_pts = [lst[i] for i in range(len(lst)) if i%2 != 0]
im1_pts = np.array(im1_pts)
im2_pts = np.array(im2_pts)

im1 = np.array(Image.open('baby.jpg'))
im2 = np.array(Image.open('cat.jpg'))

warp_frac=np.arange(0,1,1/50)
dissolve_frac = warp_frac
morphed_im=morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)
imgs=[]
for i in range(0,warp_frac.shape[0]):
    imgs.append(morphed_im[i, :, :, :])
imageio.mimsave('./output.gif', imgs)