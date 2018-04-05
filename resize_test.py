import io
from PIL import Image
#import matplotlib.pyplot as plt
from torchvision import models, transforms
#from torch.autograd import Variable

#from models import Generator, Discriminator, FeatureExtractor

upsample_factor = 2

image_pil = Image.open("/home/tomron27@st.technion.ac.il/PyTorch-SRGAN-master/test.png")
# trans = transforms.ToPILImage()
# trans1 = transforms.ToTensor()

resize_tarns = transforms.Resize([image_pil.height/upsample_factor, image_pil.width/upsample_factor])

small_pil = resize_tarns(image_pil)

small_pil.save("test_small.png")

x=0