import json
import numpy
import os
import shutil
import torch
import torchvision

from torchvision import transforms

from PIL import Image
import torch.utils.data as data
import sys,time
from dataset import  InfiniteSamplerWrapper
import numpy as np



def test(model, dataloader=None):
	model.eval()
	cnt, ACC, correct_top5 = 0.0, 0, 0
	with torch.no_grad():
		for i,(img, iden) in enumerate(dataloader):
			img, iden = img.cuda(), iden.cuda()

			bs = img.size(0)
			iden = iden.view(-1)
			_,out_prob = model(img)
			out_iden = torch.argmax(out_prob, dim=1).view(-1)
			ACC += torch.sum(iden == out_iden).item()
			_, top5 = torch.topk(out_prob,5, dim = 1)  
			for ind,top5pred in enumerate(top5):
				if iden[ind] in top5pred:
					correct_top5 += 1
		
			cnt += bs
	return ACC * 100.0 / cnt,correct_top5* 100.0 / cnt

def decision(inputs,net):
	
	_,teacher_outputs = net(inputs)
	_, teacher_predicted = teacher_outputs.max(1)
	return teacher_predicted


class Tee(object):
	def __init__(self, name, mode):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		sys.stdout = self
	def __del__(self):
		sys.stdout = self.stdout
		self.file.close()
	def write(self, data):
		if not '...' in data:
			self.file.write(data)
		self.stdout.write(data)
		self.flush()
	def flush(self):
		self.file.flush()

def load_json(json_file):
	with open(json_file) as data_file:
		data = json.load(data_file)
	return data

def get_deprocessor(size=112):
	# resize 112,112
	proc = []
	proc.append(transforms.Resize((size, size)))
	proc.append(transforms.ToTensor())
	return transforms.Compose(proc)

def low2high(img):
	# 0 and 1, 64 to 112
	bs = img.size(0)
	proc = get_deprocessor()
	img_tensor = img.detach().cpu().float()
	img = torch.zeros(bs, 3, 112, 112)
	for i in range(bs):
		img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
		img_i = proc(img_i)
		img[i, :, :, :] = img_i[:, :, :]
	
	img = img.cuda()
	return img
class ImageFolder(data.Dataset):
	def __init__(self, args, file_path, mode,attackid=-1):
		self.args = args
		self.mode = mode
		self.img_size = args['dataset']['img_size']
		# print('mode',mode)
		# exit()
		if mode == 'gan':
			self.img_path = args["dataset"]["img_pub_path"]
			self.dataset = self.args["dataset"]["d_pub"]
			filename = "Public{}.npy".format(self.dataset)
		

		elif mode == 'test':
			self.img_path = args["dataset"]["img_priv_path"]
			self.dataset = self.args["dataset"]["d_priv"]
			# self.filename = self.dataset+'.npy'
			filename = "Test{}.npy".format(self.dataset)
		elif mode == 'train':
			self.img_path = args["dataset"]["img_priv_path"]
			self.dataset = self.args["dataset"]["d_priv"]
			# self.filename = self.dataset+'.npy'
			filename = "Train{}.npy".format(self.dataset)
			
		print('-------',self.dataset,filename)
		self.name_list, self.label_list = self.get_list(file_path) 
		if attackid>0:
			label = np.array(self.label_list)
			image_list = np.array(self.image_list)
			index =  label < attackid
			self.label_list = label[index]
			self.image_list = image_list[index]

		self.transform = self.get_processor()

		if os.path.isfile(filename):
			loaded_data = np.load(filename, allow_pickle=True)
			self.label_list = loaded_data.item().get('y')
			self.images = loaded_data.item().get('x')


			print(len(self.images))
		else:
			self.images = []
			for i in range(len(self.name_list)):
				img_path = self.img_path + "/" +  self.name_list[i]
				print(i,len(self.name_list))
				img = Image.open(img_path)
				if self.transform != None:
					img = self.transform(img)
				self.images.append(img)
			np.save(filename,{'x':self.images,'y':self.label_list})

		# self.image_list = self.load_img()
		self.num_img = len(self.images)
		self.n_classes = args["dataset"]["n_classes"]
		if self.mode is not "gan":
			print("Load " + str(self.num_img) + " images")
		
		
	def get_list(self, file_path):
		name_list, label_list = [], []
		f = open(file_path, "r")
		for line in f.readlines():
			if self.mode == "gan":
				img_name = line.strip()

			else:
				# print(' line', line.strip().split(' '))
				img_name, iden = line.strip().split(' ')
				label_list.append(int(iden))
			name_list.append(img_name)
		return name_list, label_list

	
	def load_img(self):
		img_list = []
		
		for i, img_name in enumerate(self.name_list):
			if img_name.endswith(".png") or  img_name.endswith(".jpg") or  img_name.endswith(".jpeg") :
				path = self.img_path + "/" + img_name
				img = PIL.Image.open(path)
				# img = processer(img.convert('RGB'))
				img_list.append(img)#.copy())
				# img.close()
								
		return img_list
	
	
	def get_processor(self):
		# if self.model_name in ("FaceNet", "FaceNet_all"):
		# 	re_size = 112
		# else:
		re_size = self.img_size

		if 'celeba' in self.dataset:
			crop_size = 108
			offset_height = (218 - crop_size) // 2
			offset_width = (178 - crop_size) // 2
		elif self.dataset == 'facescrub':
			crop_size = 108
			offset_height = (218 - crop_size) // 2
			offset_width = (178 - crop_size) // 2
		elif self.dataset == 'ffhq':
			# print('-------ffhq')
			crop_size = 88 #88
			offset_height = (128 - crop_size) // 2 
			offset_width = (128 - crop_size) // 2
		elif self.dataset == 'pubfig':
			
			crop_size = 67
			offset_height = (100 - crop_size) // 2
			offset_width = (100 - crop_size) // 2
				
		crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

		proc = []
		# if self.mode == "train":
		# 	proc.append(transforms.ToTensor())
		# 	proc.append(transforms.Lambda(crop))
		# 	proc.append(transforms.ToPILImage())
		# 	proc.append(transforms.Resize((re_size, re_size)))
		# 	proc.append(transforms.RandomHorizontalFlip(p=0.5))
		# 	proc.append(transforms.ToTensor())
		# else:			
		# 	proc.append(transforms.ToTensor())
		# 	if self.args["dataset"]["d_priv"] != 'facescrub':
		# 		proc.append(transforms.Lambda(crop))

		# 	proc.append(transforms.ToPILImage())
		# 	proc.append(transforms.Resize((re_size, re_size)))
		# 	proc.append(transforms.ToTensor())
		
		if self.mode == "train":
				
			proc.append(transforms.ToTensor())
			
			if self.dataset != 'vggface2' and self.dataset != 'facescrub':
				proc.append(transforms.Lambda(crop))
			proc.append(transforms.ToPILImage())
			
			if self.dataset != 'vggface2' and self.dataset != 'facescrub':
				proc.append(transforms.Resize((re_size, re_size)))
			proc.append(transforms.RandomHorizontalFlip(p=0.5))
			proc.append(transforms.ToTensor())
		else:
			proc.append(transforms.ToTensor())
			
			if self.dataset != 'vggface2' and self.dataset != 'facescrub':
				proc.append(transforms.Lambda(crop))
			proc.append(transforms.ToPILImage())
			
			if self.dataset != 'vggface2' and self.dataset != 'facescrub':
				proc.append(transforms.Resize((re_size, re_size)))
			proc.append(transforms.ToTensor())

		return transforms.Compose(proc)

	def __getitem__(self, index):
				
		# processer = self.get_processor()
		# img = processer(self.image_list[index])
		img = self.images[index]
		if self.mode == "gan":
			return img
		label = self.label_list[index]

		return img, label#,self.name_list[index]

	def __len__(self):
		return self.num_img


# import  PIL
# class ImageFolder(data.Dataset):
#	 def __init__(self, args, file_path, mode,attackid =-1):
#		 self.args = args
#		 self.mode = mode
#		 # self.img_path = args["dataset"]["img_path"]
#		 self.img_size = args['dataset']['img_size']
#		 if mode == 'gan':
#			 self.img_path = args["dataset"]["img_pub_path"]
#			 self.dataset = self.args["dataset"]["d_pub"]
#		 else:
#			 self.img_path = args["dataset"]["img_priv_path"]
#			 self.dataset = self.args["dataset"]["d_priv"]

#		 self.processor = self.get_processor()
#		 self.image_list, self.label_list = self.get_list(file_path)
#		 if attackid>0:
#			 label = np.array(self.label_list)
#			 image_list = np.array(self.image_list)
#			 index =  label < attackid
#			 self.label_list = label[index]
#			 self.image_list = image_list[index]

#		 self.num_img = len(self.image_list)
#		 self.n_classes = args["dataset"]["n_classes"]
#		 if self.mode is not "gan":
#			 print("Load " + str(self.num_img) + " images")

#	 def get_list(self, file_path):
#		 name_list, label_list = [], []
#		 f = open(file_path, "r")
#		 for line in f.readlines():
#			 if self.mode == "gan":
#				 img_name = line.strip()
#			 else:
#				 img_name, iden = line.strip().split(' ')
#				 label_list.append(int(iden))
#			 name_list.append(img_name)

#		 return name_list, label_list

#	 def load_img(self, path):
#		 # data_root = self.args["dataset"]["img_path"]
#		 img = PIL.Image.open(os.path.join(self.img_path , path))

#		 img = img.convert('RGB')

#		 return img

#	 def get_processor(self):
#		 re_size = self.img_size
#		 # elif self.args['dataset']['name'] == "cifar":
#		 #	 re_size = 32
#		 # else:
#		 #	 re_size = 64

#		 if self.dataset  == "celeba":
#			 crop_size = 108
#			 offset_height = (218 - crop_size) // 2
#			 offset_width = (178 - crop_size) // 2

#		 # elif self.args['dataset']['name'] == "facescrub":
#		 #	 # NOTE: dataset face scrub
#		 #	 crop_size = 64
#		 #	 offset_height = (64 - crop_size) // 2
#		 #	 offset_width = (64 - crop_size) // 2

		
#		 elif self.dataset == 'facescrub':
#			 crop_size = 108
#			 offset_height = (218 - crop_size) // 2
#			 offset_width = (178 - crop_size) // 2
#		 elif self.dataset == 'ffhq':
#			 # print('-------ffhq')
#			 crop_size = 88 #88
#			 offset_height = (128 - crop_size) // 2 
#			 offset_width = (128 - crop_size) // 2
#		 elif self.dataset == 'pubfig':
#			 crop_size = 67
#			 offset_height = (100 - crop_size) // 2
#			 offset_width = (100 - crop_size) // 2

		
#		 crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

#		 proc = []
#		 if self.mode == "train":
				
#			 proc.append(transforms.ToTensor())
			
#			 if self.dataset != 'vggface2':
#				 proc.append(transforms.Lambda(crop))
#			 proc.append(transforms.ToPILImage())
			
#			 if self.dataset != 'vggface2':
#				 proc.append(transforms.Resize((re_size, re_size)))
#			 proc.append(transforms.RandomHorizontalFlip(p=0.5))
#			 proc.append(transforms.ToTensor())
#		 else:
#			 # if not (self.args["dataset"]["d_priv"] == 'facescrub' and self.mode != 'gan'):	
#			 #	 if self.args["dataset"]["d_pub"] != 'facescrub' and self.mode == 'gan':			 
#			 #		 proc.append(transforms.Lambda(crop))
#			 # else:
#			 # proc.append(transforms.Lambda(crop))


#			 proc.append(transforms.ToTensor())
			
#			 if self.dataset != 'vggface2':
#				 proc.append(transforms.Lambda(crop))
#			 proc.append(transforms.ToPILImage())
			
#			 if self.dataset != 'vggface2':
#				 proc.append(transforms.Resize((re_size, re_size)))
#			 proc.append(transforms.ToTensor())

#		 return transforms.Compose(proc)

#	 def __getitem__(self, index):
#		 processer = self.get_processor()
#		 img = processer(self.load_img(self.image_list[index]))
#		 if self.mode == "gan":
#			 return img

#		 label = self.label_list[index]
#		 return img, label

#	 def __len__(self):
#		 return self.num_img
	
def init_dataloader(args, file_path, batch_size=64, mode="gan", attackid = -1, iterator=False):
  
	tf =time.time()
	
	 
	data_set = ImageFolder(args, file_path, mode,attackid)

	if mode =='gan' or mode =='gan_cond' or mode =='gan_cond_da':
		data_loader = iter(torch.utils.data.DataLoader(
							data_set, batch_size,
							sampler=InfiniteSamplerWrapper(data_set)))
					
	else:
		
		data_loader = torch.utils.data.DataLoader(
				data_set, batch_size,  
				generator=torch.Generator(device='cuda'),
				shuffle=True,
				drop_last=True)
				
		interval = time.time() - tf
		print('Initializing data loader took %ds' % interval)
	
	return data_set, data_loader

class Dict2Args(object):
	"""Dict-argparse object converter."""

	def __init__(self, dict_args):
		for key, value in dict_args.items():
			setattr(self, key, value)


def generate_images(gen, device, batch_size=64, dim_z=128, distribution=None,
					num_classes=None, class_id=None):
	"""Generate images.

	Priority: num_classes > class_id.

	Args:
		gen (nn.Module): generator.
		device (torch.device)
		batch_size (int)
		dim_z (int)
		distribution (str)
		num_classes (int, optional)
		class_id (int, optional)

	Returns:
		torch.tensor

	"""

	z = sample_z(batch_size, dim_z, device, distribution)
	if num_classes is None and class_id is None:
		y = None
	elif num_classes is not None:
		y = sample_pseudo_labels(num_classes, batch_size, device)
	elif class_id is not None:
		y = torch.tensor([class_id] * batch_size, dtype=torch.long).to(device)
	else:
		y = None
	with torch.no_grad():
		fake = gen(z, y)

	return fake


def sample_z(batch_size, dim_z, device, distribution=None):
	"""Sample random noises.

	Args:
		batch_size (int)
		dim_z (int)
		device (torch.device)
		distribution (str, optional): default is normal

	Returns:
		torch.FloatTensor or torch.cuda.FloatTensor

	"""

	if distribution is None:
		distribution = 'normal'
	if distribution == 'normal':
		return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
	else:
		return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).uniform_()



def sample_pseudo_labels(num_classes, batch_size, device):
	"""Sample pseudo-labels.

	Args:
		num_classes (int): number of classes in the dataset.
		batch_size (int): size of mini-batch.
		device (torch.Device): For compatibility.

	Returns:
		~torch.LongTensor or torch.cuda.LongTensor.

	"""

	pseudo_labels = torch.from_numpy(
		numpy.random.randint(low=0, high=num_classes, size=(batch_size))
	)
	pseudo_labels = pseudo_labels.type(torch.long).to(device)
	return pseudo_labels


def save_images(n_iter, count, root, train_image_root, fake, real):
	"""Save images (torch.tensor).

	Args:
		root (str)
		train_image_root (root)
		fake (torch.tensor)
		real (torch.tensor)

	"""

	fake_path = os.path.join(
		train_image_root,
		'fake_{}_iter_{:07d}.png'.format(count, n_iter)
	)
	# real_path = os.path.join(
	#	 train_image_root,
	#	 'real_{}_iter_{:07d}.png'.format(count, n_iter)
	# )
	torchvision.utils.save_image(
		fake, fake_path, nrow=20, normalize=True, scale_each=True
	)
	shutil.copy(fake_path, os.path.join(root, 'fake_latest.png'))
	# torchvision.utils.save_image(
	#	 real, real_path, nrow=4, normalize=True, scale_each=True
	# )
	# shutil.copy(real_path, os.path.join(root, 'real_latest.png'))


def save_checkpoints(args, n_iter, count, gen, opt_gen, dis, opt_dis):
	"""Save checkpoints.

	Args:
		args (argparse object)
		n_iter (int)
		gen (nn.Module)
		opt_gen (torch.optim)
		dis (nn.Module)
		opt_dis (torch.optim)

	"""

	count = n_iter // args.checkpoint_interval
	gen_dst = os.path.join(
		args.results_root,
		'gen_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
	)
	torch.save({
		'model': gen.state_dict(), 'opt': opt_gen.state_dict(),
	}, gen_dst)
	shutil.copy(gen_dst, os.path.join(args.results_root, 'gen_latest.pth.tar'))
	dis_dst = os.path.join(
		args.results_root,
		'dis_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
	)
	torch.save({
		'model': dis.state_dict(), 'opt': opt_dis.state_dict(),
	}, dis_dst)
	shutil.copy(dis_dst, os.path.join(args.results_root, 'dis_latest.pth.tar'))


def resume_from_args(args_path, gen_ckpt_path, dis_ckpt_path):
	"""Load generator & discriminator with their optimizers from args.json.

	Args:
		args_path (str): Path to args.json
		gen_ckpt_path (str): Path to generator checkpoint or relative path
							 from args['results_root']
		dis_ckpt_path (str): Path to discriminator checkpoint or relative path
							 from args['results_root']

	Returns:
		gen, opt_dis
		dis, opt_dis

	"""

	from models.generators import resnet64
	from models.discriminators import snresnet64

	with open(args_path) as f:
		args = json.load(f)
	conditional = args['cGAN']
	num_classes = args['num_classes'] if conditional else 0
	# Initialize generator
	gen = resnet64.ResNetGenerator(
		args['gen_num_features'], args['gen_dim_z'], args['gen_bottom_width'],
		num_classes=num_classes, distribution=args['gen_distribution']
	)
	opt_gen = torch.optim.Adam(
		gen.parameters(), args['lr'], (args['beta1'], args['beta2'])
	)
	# Initialize discriminator
	if args['dis_arch_concat']:
		dis = snresnet64.SNResNetConcatDiscriminator(
			args['dis_num_features'], num_classes, dim_emb=args['dis_emb']
		)
	else:
		dis = snresnet64.SNResNetProjectionDiscriminator(
			args['dis_num_features'], num_classes
		)
	opt_dis = torch.optim.Adam(
		dis.parameters(), args['lr'], (args['beta1'], args['beta2'])
	)
	if not os.path.exists(gen_ckpt_path):
		gen_ckpt_path = os.path.join(args['results_root'], gen_ckpt_path)
	gen, opt_gen = load_model_optim(gen_ckpt_path, gen, opt_gen)
	if not os.path.exists(dis_ckpt_path):
		dis_ckpt_path = os.path.join(args['results_root'], dis_ckpt_path)
	dis, opt_dis = load_model_optim(dis_ckpt_path, dis, opt_dis)
	return Dict2Args(args), gen, opt_gen, dis, opt_dis


def load_model_optim(checkpoint_path, model=None, optim=None):
	"""Load trained weight.

	Args:
		checkpoint_path (str)
		model (nn.Module)
		optim (torch.optim)

	Returns:
		model
		optim

	"""

	ckpt = torch.load(checkpoint_path)
	if model is not None:
		model.load_state_dict(ckpt['model'])
	if optim is not None:
		optim.load_state_dict(ckpt['opt'])
	return model, optim


def load_model(checkpoint_path, model):
	"""Load trained weight.

	Args:
		checkpoint_path (str)
		model (nn.Module)

	Returns:
		model

	"""

	return load_model_optim(checkpoint_path, model, None)[0]


def load_optim(checkpoint_path, optim):
	"""Load optimizer from checkpoint.

	Args:
		checkpoint_path (str)
		optim (torch.optim)

	Returns:
		optim

	"""

	return load_model_optim(checkpoint_path, None, optim)[1]


def save_tensor_images(images, filename, nrow=None, normalize=True):
	if not nrow:
		torchvision.utils.save_image(images, filename, normalize=normalize, padding=0)
	else:
		torchvision.utils.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)
