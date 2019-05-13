import argparse
import utils
import model_utils

parser = argparse.ArgumentParser(description='Make a prediction using a deep learning model!')
parser.add_argument('image', help='pass in the path to the image for which you want a prediction', nargs=1)
parser.add_argument('checkpoint', help='load a pretrained model using a checkpoint file', nargs=1)
parser.add_argument('--gpu', default=False, action='store_true', help='add this flag to predict using the GPU!')
parser.add_argument('--topk', default=1, type=int, nargs='?')
parser.add_argument('--category_names', nargs='?', default=None, help='specify category names')
args = parser.parse_args()

print("Loading and transforming image...")
image = utils.process_image(args.image[0])
print("Image loaded!")

print("Loading checkpoint...")
model = model_utils.load_checkpoint(args)
print("Checkpoint loaded!")
top_p, output_classes, output_names = model_utils.predict(model, image, args)
print("Making prediction...")

print("The probabilities are as follows:")
if output_names:
    for i in range(len(top_p)):
        print(f"There is a {round(top_p[i]*100)}% chance that it is a {output_names[i]} (id: {output_classes[i]})")
else:
    for i in range(len(top_p)):
        print(f"There is a {round(top_p[i]*100)}% chance that it is an item with class {output_classes[i]}")    

