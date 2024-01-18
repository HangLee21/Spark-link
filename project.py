import cv2
import numpy as np
import json
import yaml


name_dict = ['project_front', 'project_back', 'project_left', 'project_right']

def represent_list(dumper, data):
    # 将列表以[]形式输出
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def main():
    #read the undistort image
    img_path='/home/sparklink/Downloads/Spark-link/demarcate/undistorted_image_'
    trans_img_path='/home/sparklink/Downloads/Spark-link/demarcate/projected_image_'
    
    proj_file='/home/sparklink/Downloads/Spark-link/demarcate/projection.json'
    proj_params={}

    for i in range(0,4):
        img = cv2.imread(img_path+str(i)+'.jpg')

        #get height and width
        h,w,channes=img.shape
        print('h=',h,'w=',w)
        #find the key point
        ratio=1280/10
        a=175

        jsonfile='./location.json'
        with open(jsonfile,'r') as f:
            location=json.load(f)
        
        #print(location)
        
        img_loc=location[str(i)]
        proj_loc=[[(w-a)/2,(h-a)/2],
                  [(w+a)/2,(h-a)/2],
                  [(w-a)/2,(h+a)/2],
                  [(w+a)/2,(h+a)/2]]

        src=np.array(img_loc,dtype=np.float32)
        dst=np.array(proj_loc,dtype=np.float32)

        #calculate persperctive matrix
        proj_mat=cv2.getPerspectiveTransform(src,dst)
        proj_list=proj_mat.tolist()
        
        proj_params[i]=proj_list
        
        #get the transformed image
        trans_img=cv2.warpPerspective(img,proj_mat,(w,h))
        
        
        cv2.imshow(f'projected_image_{i},press ESC to continue',trans_img)
        cv2.imwrite(trans_img_path+str(i)+'.jpg',trans_img)
        
        key=cv2.waitKey(0)
        if key==27:
            #close all windows
            cv2.destroyAllWindows()

    for i in range(4):
        cmx = []
        for j in proj_params[i]:
            print(j)
            for k in j:
                cmx.append(k)
        print(cmx)
        yaml_dict = {
            "project_matrix":  {
                "rows": 3,
                "cols": 3,
                "dt": "d",
                "data": cmx,
            }
        }

        yaml.add_representer(list, representer=represent_list)
        yaml_data = yaml.dump(yaml_dict, sort_keys=False)
        # 将 JSON 写入文件
        yaml_data_with_header = "%YAML:1.0\n---\n"+yaml_data
        with open(f"./json/{name_dict[i]}.yaml", "w") as f:
            f.write(yaml_data_with_header)




    with open(proj_file,'w') as f:
        json.dump(proj_params,f)

if __name__ == "__main__":
    main()
