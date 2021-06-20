import os
import torch
from models.model import ResNet101, Efficient

def model_fn(inference_weights):
	model = ResNet101(weights=None, input_shape=(3, 224, 224), num_classes=5)
	# model = Efficient(model_name='efficientnet-b5', weights=None, input_shape=(3, 224, 224), num_classes=5)
	pretrained_dict = torch.load(inference_weights)
	single_dict = {}
	for k, v in pretrained_dict.items():
		single_dict[k[7:]] = v
	model.load_state_dict(single_dict)
	return model

def convert_model(model, target_path, input=torch.tensor(torch.rand(size=(1,3,112,112)))):
	model = torch.jit.trace(model, input)
	torch.jit.save(model,target_path)

if __name__ == '__main__':
	inference_weights = r'./models/src40-4-ep00036-val_acc@1_89.5402-val_lossFocalCosine_0.3718.pth'
	target_path = os.path.join('./models', '{}.tjm'.format(os.path.basename(inference_weights)[:-4]))

	model = model_fn(inference_weights)
	print('Load model complete')

	model.eval()
	with torch.no_grad():
		convert_model(model, target_path)
		print('Convert model complete')