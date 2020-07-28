# Fractional Laplacians - fourier
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2020

export laplacian_mask, apply_mask_fourier, laplacian_operator

###############################################################################
# Fractional laplacian implemented by 2D Discrete Fourier Transform

function laplacian_mask(model::Modelall,order::Number)
	
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
		delta = dk/10f0
		mask = (2*pi*(f.+delta)).^order	# damped by delta
	end
	
	return convert(Array{Float32,2},mask)
end

function apply_mask_fourier(q::Array,mask::Array,model::Modelall)
	n = model.n
	q_t = reshape(q,n)
	q_f = fftshift(fft(q_t))
	q_new = mask.*q_f
	q_ans = real.(ifft(ifftshift(q_new)))
	return q_ans
end

function laplacian_operator(model::Modelall,order::Number)
	n = model.n
	P = joLinearFunctionFwd_T(prod(n), prod(n),
	                             v -> vec(apply_mask_fourier(v,laplacian_mask(model,order),model)),
	                             w -> vec(apply_mask_fourier(w,laplacian_mask(model,order),model)),
	                             Float32,Float32,name="Fractional laplacian operator")
	return P
end