require 'nn'
require 'rnn'
require 'dpnn'
require 'optim'
require 'xlua'
require 'gnuplot'

opt = {
	manualSeed = 1,
	threads = 32,
	input_nc = 1,
	output_nc = 1,
	im_size = 64,
	depth = 5,
	ngf = 64,
	ndf = 96,
	lr = 0.0005,
	beta1 = 0.5,
	lambda = 1,
	num_epochs = 15,
	batch_size = 64,
	data_path = '/home/koustav.m/data/moving_mnist/moving_mnist.mat',
	disp_model = 0,
	print_freq = 2000,
	neg_val = 0.2,
	gpu = 1,
	cudnn = 1,
	save_folder = 'temp',
	save = 0,
	plot = 0,
	validate = 0,
	augment = 1
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

if opt.gpu > 0 then
	require 'cunn'
end
if opt.cudnn == 1 then
	require 'cudnn'
end
