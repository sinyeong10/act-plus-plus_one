import torch
mu = torch.randn(16, 8, 4,4)
print(mu.size(0), mu.size(1))
print(mu.shape)
print(mu.data.ndimension())
if mu.data.ndimension() == 4:
    mu = mu.view(mu.size(0), 8)#mu.size(1))
print(mu.size)
print(mu.shape)
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


# # 파일 경로 지정
# # dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus_one1\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_1.hdf5'

# dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus_one1\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_1.hdf5'

# # 파일 열기 및 구조 출력
# try:
#     with h5py.File(dataset_path, 'r') as file:
#         print("File structure:")
#         print_structure(file)
# except FileNotFoundError:
#     print("The specified file was not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")
