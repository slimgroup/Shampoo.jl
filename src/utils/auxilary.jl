export circleShape

function circleShape(h, k ,r, nrec)
	theta = LinRange(0,2*pi*(nrec-1)/nrec, nrec)
	return h .+ r*sin.(theta), k .+ r*cos.(theta)
end
