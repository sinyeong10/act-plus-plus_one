# import torch

# # 예시 데이터 생성
# all_cam_features = [torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)]
# all_cam_pos = [torch.randn(1, 2, 32, 32), torch.randn(1, 2, 32, 32)]

# # 각 리스트의 텐서를 결합
# src = torch.cat(all_cam_features, axis=3)
# pos = torch.cat(all_cam_pos, axis=3)

# # 결과 출력
# print("src shape:", src.shape)
# print("pos shape:", pos.shape)



# import torch
# mu = torch.randn(16, 8, 4,4)
# print(mu.size(0), mu.size(1))
# print(mu.shape)
# print(mu.data.ndimension())
# if mu.data.ndimension() == 4:
#     mu = mu.view(mu.size(0), 8)#mu.size(1))
# print(mu.size)
# print(mu.shape)


# import pickle

# # 파일 열기
# with open('test_tmp\config.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 데이터 출력
# print(data.keys())

# for elem_key in data.keys():
#     print(data[elem_key])

#---

# import h5py

# def print_structure(hdf5_object, indent=0):
#     """ HDF5 파일 내부의 구조를 재귀적으로 출력하는 함수 """
#     for key in hdf5_object.keys():
#         item = hdf5_object[key]
#         print('    ' * indent + key)  # 현재 객체의 이름 출력
#         if isinstance(item, h5py.Dataset):
#             print('    ' * (indent + 1) + f"Dataset, shape: {item.shape}, dtype: {item.dtype}")
#         elif isinstance(item, h5py.Group):
#             print('    ' * (indent + 1) + "Group")
#             print_structure(item, indent + 2)  # 그룹의 내부를 재귀적으로 출력
            
# def write_structure_to_file(hdf5_object, indent=0, file=None):
#     """ HDF5 파일 내부의 구조를 재귀적으로 파일에 쓰는 함수 """
#     if file is None:
#         return
    
#     for key in hdf5_object.keys():
#         item = hdf5_object[key]
#         file.write('    ' * indent + key + '\n')  # 현재 객체의 이름을 파일에 쓰기
#         if isinstance(item, h5py.Dataset):
#             file.write('    ' * (indent + 1) + f"Dataset, shape: {item.shape}, dtype: {item.dtype}\n")
#         elif isinstance(item, h5py.Group):
#             file.write('    ' * (indent + 1) + "Group\n")
#             write_structure_to_file(item, indent + 2, file)  # 그룹의 내부를 재귀적으로 파일에 쓰기

# def write_all_data(hdf5_object, file=None):
#     """ HDF5 파일의 모든 데이터를 파일에 쓰는 함수 """
#     if file is None:
#         return
    
#     for key in hdf5_object.keys():
#         item = hdf5_object[key]
#         if isinstance(item, h5py.Dataset):
#             file.write(f"Dataset '{key}' values:\n")
#             for row in item:
#                 file.write(', '.join(map(str, row)) + '\n')
#         elif isinstance(item, h5py.Group):
#             write_all_data(item, file)

# # 파일 경로 지정
# # dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_1.hdf5'

# # dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_5.hdf5'
# # output_file_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\structure5.txt'
# dataset_path = r"C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_0.hdf5"
# output_file_path = r"C:\Users\cbrnt\OneDrive\문서\act-plus-plus\structure1.txt"
# # 파일 열기 및 구조 출력
# try:
#     with h5py.File(dataset_path, 'r') as file:
#         with open(output_file_path, 'w') as output_file:
#             output_file.write("File structure:\n")
#             write_structure_to_file(file, file=output_file)

#             output_file.write("\n\nAll data:\n")
#             write_all_data(file, file=output_file)

#         print("File structure:")
#         print_structure(file)
# except FileNotFoundError:
#     print("The specified file was not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")


#---

import cv2

cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("Camera Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
