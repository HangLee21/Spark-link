import cv2

image_path='./demarcate/'

for i in range(0,4):
    path=image_path+'undistorted_image_'+str(i)+'.png'
    print(path)
    img=cv2.imread(path)


    cv2.imshow(str(i)+'0',img)
    key=cv2.waitKey(0)
    
    if key==27:
        #close all windows
        cv2.destroyAllWindows()