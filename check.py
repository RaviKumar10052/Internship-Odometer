import numpy as np
import cv2
import imutils
from skimage import exposure
from pytesseract import image_to_string
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

def edged_image(img, should_save=False):
  image=cv2.imread(img)
  image = imutils.resize(image,height=300)
  #gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
  edged_image = cv2.Canny(image, 30, 200)
  #cv2.imshow("gray", edged_image)
  #cv2.waitKey(0)


  if should_save:
    cv2.imwrite('cntr.jpg')

  return edged_image

def display_contour(edge_img):
  #cv2.imshow("Edged Image", edge_img)
  #cv2.waitKey(0)
  display_cont = None
  edge_copy = edge_img.copy()
  #cv2.imshow("Edge copy", edge_copy)
  #cv2.waitKey(0)
  contours,hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
  #print(top_cntrs)

  for cntr in top_cntrs:
    peri = cv2.arcLength(cntr,True)
    approx = cv2.approxPolyDP(cntr,0.03* peri, True)
    #print(approx)

    if len(approx) == 4:
      display_cont = approx
      break

  return display_cont

def crop_display(image):
  edge_image = edged_image(image)
  display_cont = display_contour(edge_image)
  #print(display_contour)
  cntr_pts = display_cont.reshape(4,2)
  #print(cntr_pts)
  return cntr_pts


def normalize_contrs(img,cntr_pts):
  ratio = img.shape[0] / 300.0
  norm_pts = np.zeros((4,2), dtype="float32")

  s = cntr_pts.sum(axis=1)
  norm_pts[0] = cntr_pts[np.argmin(s)]
  norm_pts[2] = cntr_pts[np.argmax(s)]

  d = np.diff(cntr_pts,axis=1)
  norm_pts[1] = cntr_pts[np.argmin(d)]
  norm_pts[3] = cntr_pts[np.argmax(d)]

  norm_pts *= ratio

  (top_left, top_right, bottom_right, bottom_left) = norm_pts

  width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
  width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
  height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
  height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

  max_width = max(int(width1), int(width2))
  max_height = max(int(height1), int(height2))

  dst = np.array([[0,0], [max_width -1, 0],[max_width -1, max_height -1],[0, max_height-1]], dtype="float32")
  persp_matrix = cv2.getPerspectiveTransform(norm_pts,dst)
  return cv2.warpPerspective(img,persp_matrix,(max_width,max_height))

def process_image(orig_image):

  img01= cv2.imread(orig_image)
  crop= crop_display(orig_image)
  display_image = normalize_contrs(img01,crop)
  #display image is now segmented.
  #cv2.imshow("disply",display_image)  9
  #cv2.waitKey(0)
  gry_disp = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
  gry_disp = exposure.rescale_intensity(gry_disp, out_range= (0,255))
  return gry_disp

def get_string(image):

    # Apply dilation and erosion to remove some noise
   # kernel = np.ones((1, 1), np.uint8)
   ## img = cv2.dilate(image, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    #cv2.imshow("after effect",img)
    #cv2.waitKey(0)



    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #im=cv2.imshow("after effect", image)
    #cv2.waitKey(0)

    try:
      cv2.imwrite("a.png",image)
    except AttributeError:
      print("Couldn't save image {}".format(image))

    x=cv2.imread("a.png")
    cv2.imshow("effect",x)
    cv2.waitKey(0)

    result =image_to_string(Image.open('a.png'))
    #print(result)
    return result

def image(orig_image):
  image= process_image(orig_image)
  text= get_string(image)
  print(text)

image("2.jpeg")
