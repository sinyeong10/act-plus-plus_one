#정확하지 않음!
#이걸로 먼저 해서 결과 보여주고 이상할 때 사용자가 처리하는 식으로!
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

visualize = False

for k in range(1, 48+1):
    filename = f"two_cam_episode_{k}_image"
    output_folder = f"scr\\mask_data\\{filename}_mask"
    os.makedirs(output_folder, exist_ok=True)  # 폴더가 없으면 생성

    # Define the red color range
    lower_red = np.array([150, 0, 0])  # Lower bound for red color
    upper_red = np.array([255, 100, 100])  # Upper bound for red color

    for i in range(115):
        all_frame_filename = [f"right_wrist_frame_{i}.jpg", f"top_frame_{i}.jpg"]
        for frame_name in all_frame_filename:
            # Load the image
            # frame_name s= f"top_frame_{i}" #right_wrist_frame_0, top_frame_0
            image_path = f'scr\\mask_data\\{filename}\\{frame_name}'
            image = Image.open(image_path)

            if image is None:
                print(f"Could not read image {image_path}")
                continue

            # Convert image to RGB
            image_rgb = image.convert('RGB')
            image_array = np.array(image_rgb)

            # Create a mask for the red areas
            mask = np.all((image_array >= lower_red) & (image_array <= upper_red), axis=-1)
            # Find connected components within the red mask to identify multiple red regions
            from scipy.ndimage import label

            # Label connected regions in the red mask
            labeled_mask, num_features = label(mask)

            # Create an output image to visualize the bounding boxes
            output_image_with_boxes = image_array.copy()

            # Create an output image array with everything set to 0 (black) initially
            bounded_image_array_multiple = np.zeros_like(image_array)

            # Iterate through each identified feature and draw a bounding box
            for j in range(1, num_features + 1):
                region = np.where(labeled_mask == j)
                min_y, max_y = np.min(region[0]), np.max(region[0])
                min_x, max_x = np.min(region[1]), np.max(region[1])
                
                # # Draw the bounding box on the visualization image
                # output_image_with_boxes[min_y:max_y+1, [min_x, max_x]] = [255, 255, 255]  # Draw vertical lines
                # output_image_with_boxes[[min_y, max_y], min_x:max_x+1] = [255, 255, 255]  # Draw horizontal lines
                
                # Copy the bounded red area to the final output image array
                bounded_image_array_multiple[min_y:max_y+1, min_x:max_x+1] = image_array[min_y:max_y+1, min_x:max_x+1]
            
            if visualize:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display the result with the bounding boxes for both red areas
                axes[0].imshow(output_image_with_boxes)
                axes[0].axis('off')
                axes[0].set_title(f"Image {i}")

                # Display the final image with areas outside the bounding boxes set to 0
                final_output_image_multiple = Image.fromarray(bounded_image_array_multiple)
                axes[1].imshow(final_output_image_multiple)
                axes[1].axis('off')
                axes[1].set_title(f"Transformed Image {i}")

                plt.show()

            # print("time")
            # import time
            # time.sleep(5)
            # plt.close()

            import cv2
            cv2.imwrite(f"{output_folder}\\{frame_name}",  cv2.cvtColor(bounded_image_array_multiple, cv2.COLOR_RGB2BGR))
            print(f"{output_folder}\\{frame_name} save")
            
            # import cv2
            # cv2.imshow("image", cv2.imread(image_path))
            # print("Press 'w' to save or 'q' to skip:")
            # while True:
            #     key = cv2.waitKey(1) & 0xFF
            #     # time.sleep(1)
            #     # print(key)
            #     if key == ord('w'):  # If 's' is pressed, save the images
            #         print(f"asd{output_folder}\\{frame_name}.jpg")
            #         final_output_image_multiple.save(f"{output_folder}\\{frame_name}.jpg")
            #         print(f"{frame_name} save")
            #         break
            #     elif key == ord('q'):  # If 'q' is pressed, skip saving
            #         print(f"{frame_name} skipped")
            #         break
        