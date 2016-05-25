require 'torch'
require 'image'

stego = {}



function stego.loadDataset(start,eend,opt,complete)



	 local nExample=eend-start+1

	 local size=nExample

	 print(complete)

-- during the learning phase, we use both cover and stego images
	 if complete then
	 		size=2*size
	 end

	 print(size)

 	 data = torch.Tensor(size,1,512,512)
	 labels=torch.ByteTensor(size)


--WARNING for jpg image, you need to change pgm by jpg

	 if complete then

	 		--training or validation phase
			for i = start,eend do
	 		 		data[2*i-2*start+1]=image.load(opt.cover.."/"..tostring(i)..opt.ext)
			 		labels[2*i-2*start+1]=1
 			 		data[2*i-2*start+1+1]=image.load(opt.stego.."/"..tostring(i)..opt.ext)
 			 		labels[2*i-2*start+1+1]=2
	 		end


	 else
			 --testing phase
	  	 for i = start,eend do
	 	  	 if i%2==0 then
	   	   		data[i-start+1]=image.load(opt.cover.."/"..tostring(i)..opt.ext)
		 	  		labels[i-start+1]=1
   		   else
				  	data[i-start+1]=image.load(opt.stego.."/"..tostring(i)..opt.ext)
						labels[i-start+1]=2
			   end
	     end

	 end			

	



 local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return size
   end


   dataset.data = dataset.data[{{1,size},{},{},{}}]
	 dataset.labels = dataset.labels[{{1,size}}]



   local labelvector = torch.zeros(2)


 setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   end})



   return dataset
end
