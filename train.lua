disp = require 'display'
disp.configure({port = 8001})
require 'image'

local function init_train(lr, beta1)

	optimState_g = {
		learningRate = lr,
		beta1 = beta1,
	}
	
	assert(net_g, "Error : Models not defined" )
	parameters_g, gradParameters_g = net_g:getParameters()

end

local function augment_data(data)
	
	local im_size = opt.im_size
	local channels = opt.input_nc
	local depth = opt.depth

	data = data:reshape(2*channels*depth, im_size, im_size)
	local trans_x = (torch.random() % 5)
	if torch.random() % 2 == 0 then
		trans_x = -trans_x
	end
	local trans_y = (torch.random() % 5)
	if torch.random() % 2 == 0 then
		trans_y = -trans_y
	end
	local degrees = (torch.random() % 4)
	if torch.random() % 2 == 0 then
		degrees = -degrees
	end
	local rotate = degrees * (math.pi / 180)
	
	local im_size = im_size

	for i = 1, data:size(1) do
		donkeys:addjob(
			tonumber((i%opt.threads) + 1),
			function()
				im = data[i]:squeeze()
				im = image.translate(im, trans_x, trans_y)
				im = image.rotate(im, rotate)
				data[i] = im
			end,
			function()
			end,
			trans_x, trans_y, rotate, im_size
		)
	end
	donkeys:synchronize()
	return data:reshape(2, 1, channels, depth, im_size, im_size)
end

local function validate(valid_data, batch_size)

	local tesize = get_size(valid_data)
	local depth = opt.depth
	local im_size = opt.im_size
	local channels = opt.input_nc
	local l1_error = 0
	
	print(color.green'\n====> Running validation')
	for t = 1, tesize, batch_size do
		xlua.progress(t, tesize)
		local num_samples = math.min(batch_size, tesize - t + 1)
	    local input_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
	    local real_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
	    local fake_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		        
    	local data = valid_data[{{}, {t, t+num_samples-1}}]:float()
		input_g:copy(data[1])
		real_output_g:copy(data[2])
	    
	    -- Scale 0 to 1
		if input_g:max() > 1 then
			input_g = scale_zero_one(input_g)
			real_output_g = scale_zero_one(real_output_g)
		end
		-- Scale -1 to 1 (tanh)
		input_g = input_g:mul(2):add(-1)
		real_output_g = real_output_g:mul(2):add(-1)

		if opt.gpu > 0 then 
			input_g = input_g:cuda()
			real_output_g = real_output_g:cuda()
			fake_output_g = fake_output_g:cuda()
		end

		fake_output_g = net_g:forward(input_g)
		local err_l1 = criterionAE:forward(fake_output_g, real_output_g)
		l1_error = l1_error + num_samples * err_l1
	end

	local avg_l1 = l1_error / tesize
	print(color.blue "\nValidation average L1 error : " .. avg_l1)
	
	return avg_l1
end

function train(train_data, num_epochs, batch_size, valid_data)

	local im_size = opt.im_size
	local channels = opt.input_nc
	local depth = opt.depth
	local err_d, err_g, err_l1 = 0, 0, 0
	local trsize = get_size(train_data)
	local epoch_tm = torch.Timer()
	local folder_path = paths.concat('/home/koustav.m/models/trial/', opt.save_folder)
	paths.mkdir(folder_path)
	local file = torch.DiskFile(paths.concat('/home/koustav.m/models/trial/', opt.save_folder, 'stats.txt'), 'w')
	if opt.plot == 1 then
		train_plot_x = {}
		train_plot_y = {}
		if opt.validate == 1 then
			valid_plot_x = {}
			valid_plot_y = {}
		end
	end
	local max_l1 = math.huge
	
	if opt.gpu > 0 then 
		
		if opt.cudnn == 1 then
      		net_g = cudnn_convert_custom(net_g, cudnn); 
		end		

		net_g:cuda();
		criterionAE:cuda();
	end
	
	init_train(opt.lr, opt.beta1)
	local idx = 0

	for epoch = 1, num_epochs do

		local counter = 0
		local l1_error = 0
		local shuffle = torch.randperm(trsize)
		epoch_tm:reset()

		for t = 1, trsize, batch_size do
			
    		xlua.progress(t % opt.print_freq, opt.print_freq)
    		local num_samples = math.min(batch_size, trsize - t + 1)
		    local input_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		    local real_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		    local fake_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		    local k = 1
		    
		    for i = t, math.min(t+batch_size-1, trsize) do
	      		if opt.augment == 1 then
	      			local data = augment_data(train_data[{{}, {shuffle[i]}}]:float())
	      			input_g[k]:copy(data[1])
	      			real_output_g[k]:copy(data[2]:index(3, torch.linspace(depth,1,depth):long()))
	      			-- real_output_g[k]:copy(data[1])
	      		else
					local data = train_data[{{}, {shuffle[i]}}]:float()
	      			input_g[k]:copy(data[1])
	      			real_output_g[k]:copy(data[2])
				end
	      		k = k+1
		    end
		    
			-- Scale 0 to 1
			if input_g:max() > 1 then
				input_g = scale_zero_one(input_g)
				real_output_g = scale_zero_one(real_output_g)
			end
			-- Scale -1 to 1 (tanh)
			input_g = input_g:mul(2):add(-1)
			real_output_g = real_output_g:mul(2):add(-1)

			if opt.gpu > 0 then 
				input_g = input_g:cuda()
				real_output_g = real_output_g:cuda()
				fake_output_g = fake_output_g:cuda()
			end
			
			fake_output_g = net_g:forward(input_g)
			
			local last_input = input_g[1]:clone()
		    last_input = deprocess(last_input:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
		    disp.image(last_input, {win=18, title='input'})
		    last_input = real_output_g[1]:clone()
		    last_input = deprocess(last_input:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
		    disp.image(last_input, {win=19, title='GT'})
			last_input = fake_output_g[1]:clone()
		    last_input = deprocess(last_input:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
		    disp.image(last_input, {win=20, title='output'})

			local fGx = function(x)
			    
			    net_g:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
			    gradParameters_g:zero()
			    
		       	-- unary loss
			    local df_do_AE = torch.zeros(fake_output_g:size())
			    if opt.gpu > 0 then 
			    	df_do_AE = df_do_AE:cuda();
			    end
		     	err_l1 = criterionAE:forward(fake_output_g, real_output_g)
		     	df_do_AE = criterionAE:backward(fake_output_g, real_output_g)
			    
			    net_g:backward(input_g, df_do_AE)
			    
			    return err_l1, gradParameters_g
			end

        	optim.adam(fGx, parameters_g, optimState_g)
        	l1_error = l1_error + num_samples * err_l1
        	counter = counter + num_samples
        	collectgarbage()

        	if counter >= opt.print_freq then
        		local file = torch.DiskFile(paths.concat('/home/koustav.m/models/trial/', opt.save_folder, 'stats.txt'), 'rw')
        		local op = "Epoch : ".. epoch.." Generator : "..err_g.." Discriminator : "..err_d.." L1: "..err_l1.."\n"
				file:seekEnd()
				file:writeString(op)
				file:close()
				print(color.magenta "\nEpoch : " .. epoch .. color.magenta " Generator error : " .. err_g 
        			.. color.magenta " Discriminator error : " .. err_d .. color.magenta " L1 error : " .. err_l1)
        		counter = counter % opt.print_freq
			end
        end

		parameters_g, gradParameters_g = nil, nil
		parameters_g, gradParameters_g = net_g:getParameters()

		local avg_l1 = l1_error / trsize
		print(color.blue "\nAverage L1 error : " .. avg_l1)
		if opt.plot == 1 then
			table.insert(train_plot_x, epoch)
			table.insert(train_plot_y, avg_l1 * 100)
			gnuplot.figure(1)
			gnuplot.title('Training L1 Loss')
			gnuplot.plot('L1 Error', torch.Tensor(train_plot_x), torch.Tensor(train_plot_y),'-')
			gnuplot.plotflush()
		end
		
		if opt.validate == 1 then
			local valid_loss = validate(valid_data, batch_size)
			if opt.plot == 1 then
				table.insert(valid_plot_x, epoch)
				table.insert(valid_plot_y, valid_loss * 100)
				gnuplot.figure(2)
				gnuplot.title('Validation L1 Loss')
				gnuplot.plot('L1 Error', torch.Tensor(valid_plot_x), torch.Tensor(valid_plot_y),'-')
				gnuplot.plotflush()
			end
			if valid_loss < max_l1 then
				max_l1 = valid_loss
				if opt.save == 1 then
					torch.save(folder_path .. '/generator.t7', net_g:clearState())
					print(color.red "Models saved")
				end
			end
		else
			if opt.save == 1 then
				torch.save(folder_path .. '/generator.t7', net_g:clearState())
				print(color.red "Models saved")
			end
		end

		idx = idx + 1
		local save_image = torch.Tensor(1, 64, 320):zero()
		
		local input = train_data[{{}, {1050}}]:clone():float()
		input = scale_zero_one(input)
		local output = input[1]:clone()
		output = output:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float()
		save_image[{{}, {}, {1, 64}}] = output[1]
		save_image[{{}, {}, {65, 128}}] = output[2]
		save_image[{{}, {}, {129, 192}}] = output[3]
		save_image[{{}, {}, {193, 256}}] = output[4]
		save_image[{{}, {}, {257, 320}}] = output[4]
		image.save('images_same/input_' .. idx .. '.png', image.scale(save_image, 320, 64))

		output:copy(input[2]:index(3, torch.linspace(depth,1,depth):long()))
		output = output:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float()
		save_image[{{}, {}, {1, 64}}] = output[1]
		save_image[{{}, {}, {65, 128}}] = output[2]
		save_image[{{}, {}, {129, 192}}] = output[3]
		save_image[{{}, {}, {193, 256}}] = output[4]
		save_image[{{}, {}, {257, 320}}] = output[4]
		image.save('images_same/GT_' .. idx .. '.png', image.scale(save_image, 320, 64))
		
		input = input:mul(2):add(-1)
		output:copy(net_g:forward(input[1]:cuda()))
		output = deprocess(output:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float()) / 255
		save_image[{{}, {}, {1, 64}}] = output[1]
		save_image[{{}, {}, {65, 128}}] = output[2]
		save_image[{{}, {}, {129, 192}}] = output[3]
		save_image[{{}, {}, {193, 256}}] = output[4]
		save_image[{{}, {}, {257, 320}}] = output[4]
		image.save('images_same/output_' .. idx .. '.png', image.scale(save_image, 320, 64))

		print(color.cyan('\nEnd of epoch %d / %d \t Time Taken: %.3f\n'):format(epoch, num_epochs, epoch_tm:time().real))
	end
end