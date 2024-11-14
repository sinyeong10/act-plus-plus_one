import h5py

def print_structure(hdf5_object, indent=0):
    """ HDF5 파일 내부의 구조를 재귀적으로 출력하는 함수 """
    for key in hdf5_object.keys():
        item = hdf5_object[key]
        print('    ' * indent + key)  # 현재 객체의 이름 출력
        if isinstance(item, h5py.Dataset):
            print('    ' * (indent + 1) + f"Dataset, shape: {item.shape}, dtype: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print('    ' * (indent + 1) + "Group")
            print_structure(item, indent + 2)  # 그룹의 내부를 재귀적으로 출력
            
def write_structure_to_file(hdf5_object, indent=0, file=None):
    """ HDF5 파일 내부의 구조를 재귀적으로 파일에 쓰는 함수 """
    if file is None:
        return
    
    for key in hdf5_object.keys():
        item = hdf5_object[key]
        file.write('    ' * indent + key + '\n')  # 현재 객체의 이름을 파일에 쓰기
        if isinstance(item, h5py.Dataset):
            file.write('    ' * (indent + 1) + f"Dataset, shape: {item.shape}, dtype: {item.dtype}\n")
        elif isinstance(item, h5py.Group):
            file.write('    ' * (indent + 1) + "Group\n")
            write_structure_to_file(item, indent + 2, file)  # 그룹의 내부를 재귀적으로 파일에 쓰기

def write_all_data(hdf5_object, file=None):
    """ HDF5 파일의 모든 데이터를 파일에 쓰는 함수 """
    if file is None:
        return
    
    for key in hdf5_object.keys():
        item = hdf5_object[key]
        if isinstance(item, h5py.Dataset):
            file.write(f"Dataset '{key}' values:\n")
            for row in item:
                file.write(', '.join(map(str, row)) + '\n')
        elif isinstance(item, h5py.Group):
            write_all_data(item, file)

# 파일 경로 지정
# dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_1.hdf5'

# dataset_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\episode_5.hdf5'
# output_file_path = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\scr\tonyzhao\datasets\sim_transfer_cube_scripted\structure5.txt'
dataset_path = r"scr\tonyzhao\datasets\next_sim_mycobot_320\two_cam_episode_17.hdf5" #r"scr\mycobot320_data\twocam_mycobot320_chunk20_1\mycobot320_model_run_image_0.hdf5"#r"mycobot320_script\all_data\cur\twocamtemp\two_cam_episode_0.hdf5"#r"twocam/two_cam_episode_0.hdf5"
output_file_path = r"base_d.txt"#r"mycobot320_script\all_data\cur\twocamtemp\two_cam_episode_0.txt"#r"twocam/two_cam_episode_0.txt"
# 파일 열기 및 구조 출력
try:
    with h5py.File(dataset_path, 'r') as file:
        with open(output_file_path, 'w') as output_file:
            output_file.write("File structure:\n")
            write_structure_to_file(file, file=output_file)

            output_file.write("\n\nAll data:\n")
            write_all_data(file, file=output_file)

        print("File structure:")
        print_structure(file)
except FileNotFoundError:
    print("The specified file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
