import cv2

image_path='./img/'

for i in range(0,4):
    path=image_path+str(i)+'_0.png'
    print(path)
    img=cv2.imread(path)


    cv2.imshow('0',img)
    key=cv2.waitKey(0)
    
    if key==27:
        #close all windows
        cv2.destroyAllWindows()