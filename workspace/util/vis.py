from cnn_layer_visualization import CNNLayerVisualization
from deep_dream import DeepDream
from gradcam import GradCam
from vanilla_backprop import VanillaBackprop
from misc_functions import preprocess_image
import torch

############# Examples #######

num_cnn_layer = 17
filter_pos = 23
class_id = 0

im_path = ''
image = cv2.imread(im_path, 1)
export_name = ''
model_path = ''

model = torch.load(model_path)

############# Examples #######

vis = CNNLayerVisualization(model, num_cnn_layer, filter_pos)
vis.visualise_layer_with_hooks()
vis.visualise_layer_without_hooks()

vis = DeepDream(model, num_cnn_layer, filter_pos, im_path)
vis.dream()

vis = GradCam(model, target_layer=num_cnn_layer)
processed_image = preprocess_image(image)
cam = vis.generate_cam(processed_image, class_id)
save_class_activation_on_image(image, cam, export_name)

vis = VanillaBackprop(model, preprocess_image(image), class_id)
grads = vis.generate_gradients()
save_gradient_images(grads, export_name+'_color')
grayscale_grads = convert_to_grayscale(grads)
save_gradient_images(grads, export_name+'_bw')
