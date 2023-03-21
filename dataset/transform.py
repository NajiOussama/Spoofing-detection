import random
import torchvision.transforms as transforms

class NonDistortedAugmentation:
    def __init__(self, image_size):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(size=self.image_size)
        ])

    def __call__(self, img):
        return self.transform(img)

def get_augmented_patches(img, image_size):
    t1 = NonDistortedAugmentation(image_size)
    t2 = NonDistortedAugmentation(image_size)
    return t1(img), t2(img)



