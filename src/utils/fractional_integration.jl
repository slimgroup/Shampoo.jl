# Fractional Laplacians - fourier
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2020

export integral_shot

###############################################################################
# Fractional integration

function integral_shot(C::joLinearFunction,d_obs::judiVector)
	d_out = deepcopy(d_obs)
	nrec = length(d_obs.geometry.xloc[1])
	nt = d_obs.geometry.nt[1]
	nsrc = d_obs.nsrc
	for i = 1:nsrc
		for j = 1:nrec
			d_out.data[i][:,j] = C*d_lin.data[i][:,j]
		end
	end
	return d_out
end

function integral_shot(C::joLinearFunction,d_obs::Array)
	d_out = deepcopy(d_obs)
	nrec = length(d_obs.geometry.xloc[1])
	nt = d_obs.geometry.nt[1]
	nsrc = d_obs.nsrc
	for i = 1:nsrc
		for j = 1:nrec
			d_out.data[i][:,j] = C*d_lin.data[i][:,j]
		end
	end
	return d_out
end
