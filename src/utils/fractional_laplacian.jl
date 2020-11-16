# Fractional Laplacians - fourier
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2020

export laplacian_mask, apply_mask_fourier, laplacian_operator

###############################################################################
# Fractional laplacian implemented by 2D Discrete Fourier Transform

function laplacian_mask(model::Model,order::Number)
	
	n = model.n
	d = model.d
	dk = 1/d[1]/n[1]
	mask = ones(Float32,n)
	center_x = (n[1]+1)/2
	center_z = (n[2]+1)/2
	
	x = reshape(collect(1:n[1]), :, 1)
	z = reshape(collect(1:n[1]), 1, :)
	
	f = dk*sqrt.((x.-center_x).^2f0.+(z.-center_z).^2f0)
	
	if order >= 0	# differentiation
		mask = (2*pi*f).^order
	else # integration - warning: null space of low frequency
		#delta = dk/10f0
		#mask = (2*pi*(f.+delta)).^order	# damped by delta
		center_x = Int(round(center_x))
		center_z = Int(round(center_z))
		mask[[1:center_x-1;center_x+1:end],:] = (2*pi*f[[1:center_x-1;center_x+1:end],:]).^order
		mask[center_x,[1:center_z-1;center_z+1:end]] = (2*pi*f[center_x,[1:center_z-1;center_z+1:end]]).^order
		mask[center_x,center_z] = 0.25*(mask[center_x+1,center_z]+mask[center_x-1,center_z]+mask[center_x,center_z+1]+mask[center_x,center_z-1])
	end
	
	return convert(Array{Float32,2},mask)
end

function apply_mask_fourier(q::Array,mask::Array,model::Model,info::Info)
	n = model.n
	nsrc = info.nsrc
	q_t = reshape(q,n[1],n[2],nsrc)
	q_ans = deepcopy(q_t)
	for i = 1:nsrc
		q_f = fftshift(fft(q_t[:,:,i]))
		q_new = mask.*q_f
		q_ans[:,:,i] = real.(ifft(ifftshift(q_new)))
	end
	return reshape(q_ans,size(q))
end

#function apply_mask_fourier(q::PhysicalParameter,mask::Array,model::Model,info::Info)
#	return apply_mask_fourier(vec(q.data),mask,model,info)
#end

function apply_mask_fourier(q::judiWeights,mask::Array,model::Model,info::Info)
	n = model.n
	nsrc = info.nsrc
	q_ans = deepcopy(q)
	for i = 1:nsrc
		q_t = q.weights[i]
		q_f = fftshift(fft(q_t))
		q_new = mask.*q_f
		q_ans.weights[i] = real.(ifft(ifftshift(q_new)))
	end
	return q_ans
end

function laplacian_operator(model::Model,order::Number,info::Info)
	n = model.n
	#if order >= 0
	if true
		P = joLinearFunctionFwd_T(prod(n), prod(n),
	                             v -> apply_mask_fourier(v,laplacian_mask(model,order),model,info),
	                             w -> apply_mask_fourier(w,laplacian_mask(model,order),model,info),
								 Float32,Float32,name="Fractional laplacian operator")
	else
		P1 = joLinearFunctionFwd_T(prod(n), prod(n),
								 v -> apply_mask_fourier(v,laplacian_mask(model,-order),model,info),
								 w -> apply_mask_fourier(w,laplacian_mask(model,-order),model,info),
								 Float32,Float32,name="Fractional laplacian operator")

		P = joLinearFunctionFwd_T(prod(n), prod(n),
								 v -> lsqr(P1,v,damp=1,verbose=true),
								 w -> lsqr(P1,w,damp=1,verbose=true),
								 Float32,Float32,name="Fractional laplacian operator")
	end

	return P
end