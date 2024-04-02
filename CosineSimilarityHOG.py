import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# A constant size of the image (width , heigth) for resizing the image when computing the histogram
imageScale = (150,150)

# The database folder
DB_folder = "Face_similarity_DB"

# Obtain a list of the image that was inside the database
database_name_list = os.listdir("Face_similarity_DB\data\Images")

# Obtain a list of the test image that was inside the database
test_in_database_name_list = os.listdir("Face_similarity_DB\\test\\Images\\inDatabase")

# Obtain a list of the test image that was not inside the database
test_not_in_database_name_list = os.listdir("Face_similarity_DB\\test\\Images\\notInDatabase")

# Define a class with all functions to perform cosineSimilarity with feature extraction from histogram of Oriented Gradients
class Similarity:
    # Initialize the class by putting the name of the test file and if the test file is inside the database or not
    # inDatabase = 1 -> The test image is not include in the database
    # inDatabase = 2 -> The test image is not include in the database
    def __init__(self, test_image, inDatabase):
        self.test_image = test_image
        self.inDatabase = inDatabase

    # Function to get the image file in the test folder
    # self.inDatabase to determine if the test image is in the database or not
    def accessTestFile(self):
        return cv2.imread(os.path.join(DB_folder, "test", "Images",
                                        "inDatabase" if (self.inDatabase) else "notInDatabase"
                                        , self.test_image), 0)
    
    # Function to get the image file in the data folder (database)
    def accessTargetFile(self, filename):
        return cv2.imread(os.path.join(DB_folder, "data", "Images", filename), 0)
    
    # Function to resize the image
    def imageResize(self, image):
        return cv2.resize(image, imageScale)
    
    # Function to get the name of the file by deleting the file type
    def fileNameDelete(self, filename):
        return filename.split(sep=".")[0] 
    
    # A histogram of Oriented Gradients (HOG) feature extraction function
    # reutrn the value of feature vector
    def compute_hog(self ,image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        # Step 1: Compute gradients
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        # Step 2: Compute gradient magnitude and orientation
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Step 3-4: Create histograms and normalize within blocks   
        histogram = np.zeros((image.shape[0] // pixels_per_cell[0], image.shape[1] // pixels_per_cell[1], orientations))

        for i in range(orientations):
            # Select pixels with orientations in the current range
            mask = np.logical_and(angle >= i * (180 / orientations), angle < (i + 1) * (180 / orientations))
            
            # Compute histogram for the selected pixels
            histogram[:, :, i] = np.sum(magnitude * mask, axis=(0, 1))

        # Step 5: Concatenate normalized block histograms to form the feature vector
        feature_vector = np.concatenate([histogram[i:i + cells_per_block[0], j:j + cells_per_block[1]].ravel()
                                        for i in range(0, histogram.shape[0], cells_per_block[0])
                                        for j in range(0, histogram.shape[1], cells_per_block[1])])

        # L2 normalization
        feature_vector /= np.linalg.norm(feature_vector)

        return feature_vector
    
    # Compare the similarity by using cosine_similarity function with feature extraction vector from two images
    def cosineSimilarity_HOG(self, vector1, vector2):
        return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    
    # Create a list of similarity percentage between the test image and all the image inside the database
    def similarityList(self):
        # An Array to store the similarity
        similarity_list = []

        test_image_file = self.accessTestFile() # Acess the image of the test file
        test_image_resize = self.imageResize(test_image_file) # resize the test file image

        # Compare the test image with all the image inside the database
        for target in database_name_list:
            target_image_file = self.accessTargetFile(target) # acess the image of the target file (image inside the datbase)
            target_image_resize = self.imageResize(target_image_file) # resize the target image
        
            # Compute the histogram of Oriented Gradients (HOG) feature extraction from both target and test image
            feature_extraction_vector_1 = self.compute_hog(target_image_resize) 
            feature_extraction_vector_2 = self.compute_hog(test_image_resize)

            similarity = self.cosineSimilarity_HOG(feature_extraction_vector_1,feature_extraction_vector_2) # Compute the similarity percentage between two histogram
            similarity_list.append(similarity) # Add similarity value to the array

        return similarity_list

    # Print a list of similarity percentage between the test image and all the image inside the database
    def printSimilarity(self):
        similarity_list = self.similarityList() # Get the list of similarity from the test and all image in database
        test_image_name = self.fileNameDelete(self.test_image) # Get the name of without the file type of test image

        print("\n <--- Similarity List --->\n")

        # Check the lenght of both similarity and list of image in database array
        if len(similarity_list) == len(database_name_list):
            for i in range(len(similarity_list)):
                similarity = similarity_list[i] # Get the each similarity value
                target = database_name_list[i] # Get the file of the target image
                target_image_name = self.fileNameDelete(target) # Get the name of without the file type of target image
                print("[Similarity] {} <-> {} : {:.4f}".format(test_image_name, target_image_name, similarity))

    def similarityResult(self):
        similarity_list = self.similarityList() # Get the list of similarity from the test and all image in database
        max_value = max(similarity_list)  # Find the highest value in the similarity list
        max_index = similarity_list.index(max(similarity_list)) # Find the index of the highest value in the similarity list
        max_name = database_name_list[max_index] # Find the target file name with the index of the highest value in the similarity list

        if self.inDatabase:
            result_name = self.fileNameDelete(max_name)
            print(f"\nResult : The given test image which is inside the database is the picture of {result_name}")

            # Display the image of the person
            img = cv2.imread(os.path.join(DB_folder, "data", "Images", max_name))

            # Create a subplot
            ax = plt.subplot(1,1,1)
            
            # Display the first image on the subplot
            ax.imshow(img[:,:,::-1])
            ax.set_title(result_name)

            # Hide the axes labels for better presentation
            ax.axis('off')

            # Show the plot
            plt.show()
        else:
            result_name_test = self.fileNameDelete(self.test_image)
            result_name_target = self.fileNameDelete(max_name)
            print(f"\nResult : The given test image of {result_name_test} which is not inside the database looks similar to {result_name_target} with the {max_value:.4f} % similarity percentage")

            # Display the test image and the target image with the highest similarity
            img1 = cv2.imread(os.path.join(DB_folder, "data", "Images", max_name))
            img2 = cv2.imread(os.path.join(DB_folder, "test", "Images", "notInDatabase", self.test_image))

            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display the first image on the left subplot
            axes[0].imshow(img2[:,:,::-1])
            axes[0].set_title(f"Test Image : {result_name_test}")

            # Display the second image on the right subplot
            axes[1].imshow(img1[:,:,::-1])
            axes[1].set_title(f"Target Image : {result_name_target}")

            # Hide the axes labels for better presentation
            for ax in axes:
                ax.axis('off')

            # Show the plot
            plt.show()

        
