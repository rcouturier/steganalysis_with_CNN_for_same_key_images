----------------------------------------------------------------------
-- This script shows how to train different models on the steganalysis
--
-- Many parts are based from previous works of Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset'
require 'pl'
require 'image'


----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -o,--optimization  (default "SGD")       optimization: SGD
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   -p,--type          (default float)       float or cuda
   --cover            (default "")          directory of cover images
   --stego            (default "")          directory of stego images
   --start_train      (default 1)           index of first image for train
   --end_train	      (default 1000)        index of last image for train
   --start_test       (default 1)           index of first image for test
   --end_test	      (default 1000)        index of last image for test
   --ext	      (default ".pgm")      extention of the image	 
   -d,--devid         (default 1)           device ID (if using CUDA)

]]




-- fix seed
torch.manualSeed(331)


-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())


if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
   cutorch.manualSeed(331)
--   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end




----------------------------------------------------------------------
-- define model to train
-- on the 2-class classification problem
--
classes = {'1','2'}

-- geometry: width and height of input images
geometry = {512,512}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

	       model:add(nn.SpatialConvolutionMM(1, 1, 3, 3))
      model:add(nn.Tanh())

      model:add(nn.SpatialConvolutionMM(1, 16*4, 509, 509))
      model:add(nn.Tanh())
   
      model:add(nn.Reshape(64*4))
  		model:add(nn.Linear(64*4, #classes))
															
		  model:add(nn.LogSoftMax())

      
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
   model=model:float()

end





print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--

criterion = nn.ClassNLLCriterion()



if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()


--training and testing
if opt.network == '' then

	 -- create training set and normalize
	 trainData = stego.loadDataset(opt.start_train,opt.end_train,opt,true)
	 trainData:normalizeGlobal(mean, std)


	 -- create test set and normalize
	 testData = stego.loadDataset(opt.start_test,opt.end_test,opt,false)
 	 testData:normalizeGlobal(mean, std)
else

	 -- validation set and normalize
	 -- use of an existing trained network
	 testData = stego.loadDataset(opt.start_test,opt.end_test,opt,true)
 	 testData:normalizeGlobal(mean, std)

end


----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1


   -- number of data
   local train_nrow = dataset:size(1)
	 -- shuffle at each epoch
   local shuffle = torch.randperm(train_nrow)

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)


      
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[shuffle[i]]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      if opt.type == 'cuda' then 
	 inputs = inputs:cuda()
	 targets = targets:cuda()
      end


      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:

         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
				}
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'stego_100_'..epoch..'.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
 	 torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()



   -- number of data
   local train_nrow = dataset:size(1)
	 -- shuffle at each epoch
   local shuffle = torch.randperm(train_nrow)


   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
--      -- disp progress
--      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)



      
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[shuffle[i]]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      if opt.type == 'cuda' then 
	 inputs = inputs:cuda()
	 targets = targets:cuda()
      end

      
      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end




if opt.network == '' then

	 --training and testing


	 while true do
	 -- train/test
	    train(trainData)
	    test(testData)


	    if opt.plot then
     	        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
		testLogger:style{['% mean class accuracy (test set)'] = '-'}
      		trainLogger:plot()
      		testLogger:plot()
   	    end
         end
else
	 print("Validation with a pretrained network")
	 test(testData)
end
