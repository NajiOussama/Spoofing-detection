import random
from torchvision.transforms import functional as F

class RandomResizeFlip:
    def __init__(self, resize_range=(256, 512)):
        self.resize_range = resize_range
        
    def __call__(self, img):
        # Resize
        size = random.randint(self.resize_range[0], self.resize_range[1])
        img = F.resize(img, (size, size))
        
        # Flip horizontally
        if random.random() > 0.5:
            img = F.hflip(img)
            
        return img