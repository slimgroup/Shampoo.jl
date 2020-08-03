# Fractional Laplacians - fourier
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

export RightPrecondF

###############################################################################
# Right precontioners for system F*P1*q = d  or J*P2*dm = F*q-F0*q
# where P1 and P2 are fractional Laplacians

function RightPrecondF(F::judiPDEextended)
	
	model = F.model
	info = F.info
	order = 1
	P = laplacian_operator(model::Modelall,order::Number,info::Info)
	return P
end

function RightPrecondJ(J::judiJacobian)
	
	model = J.model
	info = J.info
	order = -0.5
	P = laplacian_operator(model::Modelall,order::Number,info::Info)
	return P
end