#AUTOR: Lalykin Oleg, group:381503-3
import cv2
import numpy as np

#GRAY
def gray(img):
 gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 return gray

#CONTRAST
def contr(img): 
 contrast = cv2.equalizeHist(gray(img))
 return contrast

#EDGE
def edge(img):
 edge = cv2.Canny(contr(img), 40, 200)
 return edge

#POINTS
def cor(img):
 cor_img = edge(img)
 corners = cv2.goodFeaturesToTrack(cor_img, 25,0.01,10);
 color= [255, 255, 100, 255]
 corners = np.int0(corners)
 for i in corners:
    x,y = i.ravel()
    cv2.circle(cor_img,(x,y),10,color)
 return cor_img

def cor_s(img):
 cor_img = contr(img)
 corners = cv2.goodFeaturesToTrack(cor_img, 25,0.01,10);
 color= [255, 255, 100, 255]
 corners = np.int0(corners)
 for i in corners:
    x,y = i.ravel()
    cv2.circle(cor_img,(x,y),10,color)
 return cor_img
   
#DISTANCE
def dis(img):
 dist = cor(img)
 dist = cv2.bitwise_not(dist)
 dist = cv2.distanceTransform(dist, cv2.DIST_L2, 3)
 dist = cv2.normalize(dist,None,0.,1.,cv2.NORM_MINMAX)
 return dist

def dis_s(img):
 dist = cor(img)
 dist = cv2.bitwise_not(dist)
 dist = cv2.distanceTransform(dist, cv2.DIST_L2, 3)
 return dist

#FILTER
def Borders(c, left, right):
    if c <= left:
        return left
    if c > right:
        return right
    return c

def fil(img):
 dist = dis_s(img)   
 h,w = img.shape[0:2]
 k = 0.75
 Int = cv2.integral(contr(img))
 
 fil = np.zeros((h, w), np.uint8)

 x=0
 y=0
 for x in range(h):
    for y in range(w):
        r = min(int(k * dist[x, y]), 5)
        sh = h - 1
        sw = w - 1
        fx = x + r + 1
        fy = y + r + 1
        sx = x - r
        sy = y - r
        sbordX = Borders(sx, 0, sh)
        sbordY = Borders(sy, 0, sw)
        fbordX = Borders(fx, 0, sh)
        fbordY = Borders(fy, 0, sw)
        ch = Int[sbordX, sbordY] + Int[fbordX, fbordY] -Int[sbordX, fbordY] - Int[fbordX, sbordY]
        fil[x, y] = ch/((1 + 2 * r) ** 2)
 return fil

#READ
img = cv2.imread('pr.jpg', 1)
img = cv2.resize(img, (0,0), fx=0.9, fy=0.9) 
cv2.imshow("original", img)

cv2.imshow("gray", gray(img))
cv2.imshow("contr", contr(img))
cv2.imshow("edge", edge(img))
cv2.imshow("cor", cor_s(img))
cv2.imshow("dis", dis(img))
cv2.imshow("fil", fil(img))

cv2.waitKey() 
