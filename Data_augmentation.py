from Data_utils import *
import os

def resizeImages():
    data_root = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/GP/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/*/*'))
    all_video_paths = np.array(all_video_paths, dtype=np.str)

    for path in all_video_paths:
        path = path + '/*.jpg'
        frame_num = 0
        for file_name in sorted(glob.glob(path)):
            #print(file_name)
            img = cv2.imread(file_name)
            resized_img = cv2.resize(img, (224, 224))

            cv2.imwrite(path[:-5] + 'frame'+str(frame_num) + '.jpg', resized_img)
            os.remove(file_name)
            frame_num += 1

def flip_images():
    data_root = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/GP/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/*/*'))
    all_video_paths = np.array(all_video_paths, dtype=np.str)

    path_num = 0
    for path in all_video_paths:
        new_path = path[:-4] + '/flipped' + str(path_num)
        os.mkdir(new_path)
        frame_num = 0
        path = path + '/*.jpg'

        for file_name in sorted(glob.glob(path)):
            img = cv2.imread(file_name)
            flipped_image = cv2.flip(img, 1)
            cv2.imwrite(new_path + '/frame'+str(frame_num)+'.jpg', flipped_image)
            frame_num += 1
        path_num +=1

def transform_images():
    data_root = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/GP/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/*/*'))
    all_video_paths = np.array(all_video_paths, dtype=np.str)

    path_num = 0
    for path in all_video_paths:

        path2 = path
        path = path.split('/')
        path = '/'.join(path[:-1])
        new_path = path + '/transform' + str(path_num)
        os.mkdir(new_path)
        frame_num = 0
        path = path + '/*.jpg'
        path2 = path2 + '/*.jpg'

        for file_name in sorted(glob.glob(path2)):
            img = cv2.imread(file_name)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            hsv += 10
            transformed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            transformed_img = np.clip(transformed_img, 0, 255)
            cv2.imwrite(new_path + '/frame'+str(frame_num)+'.jpg', transformed_img)
            print(new_path + '/frame'+str(frame_num)+'.jpg')
            frame_num += 1
        path_num +=1


data_root = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/GP/ucf_action/'
data_root = pathlib.Path(data_root)
all_video_paths = list(data_root.glob('*/*'))
all_video_paths = np.array(all_video_paths, dtype=np.str)
i = 48
for path in all_video_paths:
    new_path = '/media/khaledosama/CAF8CEC8F8CEB24D/Work/FCIS/GP/ucf_action/Lifting/SpatioTransform' + str(i)
    os.mkdir(new_path)
    i += 1
    path = path + '/*.jpg'
    j = 1



    for file_name in sorted(glob.glob(path)):

        if j < 20:
            j += 1
            continue
        img = cv2.imread(file_name)
        cv2.imwrite(new_path + '/frame'+str(j)+'.jpg', img)
        j += 1

        if j>70:
            break
