# The CosineSimilarityHOG contain the all the constants, class, and functions necessary to perform a face similarity comparison
import CosineSimilarityHOG

def main():
    # First I want to demonstrate how the program work for one image before processing all the image
   
    # Initialize the instance to perform similarity comparison
    # Change the image name correspond to if the image is in the database or not
    # 1 - inside database
    # 0 - not inside the database
    cosineSimilarityHOG = CosineSimilarityHOG.Similarity(test_image="Bruce_Lunsford.jpg", inDatabase=0)

    # Print the list of similarity by compare the test image with the images in the database
    cosineSimilarityHOG.printSimilarity()

    # if the test image is in the database, the function print the name of the person inside the image, and display the image of that person
    # if the test image is not in the database, the function print the message name of the person that looks similar to the person in the test image
    # and print the picture of that person
    cosineSimilarityHOG.similarityResult()

    # Now Compare all the test image in both inDatabase and notInDatabase folder

    # All test image that include in database
    print("\n<--- Test Images In Database --->")
    for image in  CosineSimilarityHOG.test_in_database_name_list:
        cosineSimilarityHOG = CosineSimilarityHOG.Similarity(test_image=image, inDatabase=1)
        cosineSimilarityHOG.similarityResult()

    # All test image that is not  include in database
    print("\n<--- Test Images Not In Database --->")
    for image in  CosineSimilarityHOG.test_not_in_database_name_list:
        cosineSimilarityHOG = CosineSimilarityHOG.Similarity(test_image=image, inDatabase=0)
        cosineSimilarityHOG.similarityResult()

if __name__ == "__main__":
    main()

