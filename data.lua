mattorch = require 'mattorch'

function get_size(data)
	return data:size(2)
end

function scale_zero_one(data)
	return (data / 255)
end

function deprocess(data)
	return data:add(1):div(2):mul(255)
end

function convert_tsteps(data)
	local sz = data:size(2)
	data = data:reshape(2, 2, opt.depth, sz, opt.input_nc, opt.im_size, opt.im_size)
	data = data:permute(2, 1, 4, 5, 3, 6, 7)
	return data:reshape(2, 2 * sz, opt.input_nc, opt.depth, opt.im_size, opt.im_size)
end

function get_data(data_path)

	local data = mattorch.load(data_path)

	-- Data dimensions are 20 x n x 64 x 64
	-- Need to convert to 2 x (2*n) x input_nc x depth x im_size x im_size)
	local train_data = convert_tsteps(data.train_data:permute(4, 3, 2, 1))
	local valid_data = convert_tsteps(data.valid_data:permute(4, 3, 2, 1))
	local test_data = convert_tsteps(data.test_data:permute(4, 3, 2, 1))
	
	return train_data, valid_data, test_data
end
	