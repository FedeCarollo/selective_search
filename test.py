from roi_proposals import RoiProposals
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def folder_test():
    input_folder = "img"
    output_folder = "bboxes"
    
    RoiProposals.process_folder(input_folder, output_folder)

def image_test():
    roi = RoiProposals()

    img = mpimg.imread("img/image_57540.jpg")

    bb = roi.process_image(img)

    plt.imshow(img)
    for box in bb:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=1)
        plt.gca().add_patch(rect)
    plt.show()

    

if __name__ == "__main__":
    input_folder = "img"
    output_folder = "bboxes"
    
    folder_test()
    image_test()
    
