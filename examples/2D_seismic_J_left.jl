using PyPlot, JUDI, LinearAlgebra, FFTW, JOLI
using IterativeSolvers
using SlimPlotting
using DrWatson

using Shampoo

n = (201, 201)   # (x,y,z) or (x,z)
d = (5f0, 5f0)
o = (0f0, 0f0)

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

v0 = 2f0*ones(Float32,n)
v = deepcopy(v0)
v[101,101] = 2.1f0  # dirac perturbation

# Slowness squared [s^2/km^2]
m = convert(Array{Float32,2},(1f0 ./ v).^2)	# true model
m0 = convert(Array{Float32,2},(1f0 ./ v0).^2)
dm = vec(m-m0)

model = Model(n, d, o, m; nb = 200)
model0 = Model(n, d, o, m0; nb = 200)

# Model structure

dtS = 2f0
timeS = 1600f0
nt = Int64(timeS/dtS)+1

fmin = 10f0
fmax = 40f0

wavelet = zeros(Float32, nt, 1)
wavelet[50, 1] = 1f0
wavelet = low_filter(wavelet, dtS; fmin=fmin, fmax=fmax)

################################### Source/receiver geometry ######################################

idx_wb = 30
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Set up receiver geometry

nrec = n[1]

xrec = range(0f0, stop=Float32((n[1]-1)*d[1]), length=nrec)
yrec = 0f0
zrec = range(14f0, stop=14f0, length=nrec)

# receiver sampling and recording time
timeR = timeS   # receiver recording time [ms]
dtR   = dtS 
# Set up receiver structure
nsrc = 32

xsrc = convertToCell(range(0f0, stop=Float32((n[1]-1)*d[1]), length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(9f0, stop=9f0, length=nsrc))

recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

q = judiVector(srcGeometry, wavelet)

# Set up options structure for linear operators
isic = true
fs = false
opt = Options(isic=isic, free_surface=fs)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

save_dict = @strdict isic nsrc nrec fs
name = savename(save_dict; digits=6)

figure(figsize=(20,12))
plot_velocity(v, d; new_fig=false, name="Experimental setup"); colorbar();
PyPlot.scatter(xrec,zrec,marker=".",color="red",label="receivers",s=1);
PyPlot.scatter(xsrc,zsrc,marker=".",color="white",label="sources",s=1);
legend(loc=3);
xlabel("Lateral position [m]");ylabel("Horizontal position [m]");
tight_layout()
savefig(name*"_setup.png", dpi=300);

d_obs = J*dm

figure(figsize=(20,12));
subplot(2,1,1);
PyPlot.plot(0f0:dtR:timeR, wavelet);
xlabel("time [ms]");
ylabel("amplitude");
title("wavelet in time domain");
subplot(2,1,2);
PyPlot.plot(-1f0/dtR*(nt-1)/2:1f0/dtR:1f0/dtR*(nt-1)/2,abs.(fftshift(fft(wavelet))));
xlabel("frequency [Hz]");
ylabel("amplitude");
title("wavelet in frequency domain");
tight_layout()
savefig(name*"_wavelet.png", dpi=300);

# left preconditioners
P = FractionalIntegrationOp(nt,dtR,nsrc,nrec,.5f0)

figure(figsize=(20,12))
subplot(3,3,1)
plot_sdata(d_obs.data[1], (1f0, 1f0); new_fig=false, name="J*dm");colorbar();
subplot(3,3,2)
plot_sdata(d_obs.data[4], (1f0, 1f0); new_fig=false, name="J*dm");colorbar();
subplot(3,3,3)
plot_sdata(d_obs.data[nsrc], (1f0, 1f0); new_fig=false, name="J*dm");colorbar();
subplot(3,3,4)
plot_sdata((P*d_obs).data[1], (1f0, 1f0); new_fig=false, name="P*J*dm");colorbar();
subplot(3,3,5)
plot_sdata((P*d_obs).data[4], (1f0, 1f0); new_fig=false, name="P*J*dm");colorbar();
subplot(3,3,6)
plot_sdata((P*d_obs).data[nsrc], (1f0, 1f0); new_fig=false, name="P*J*dm");colorbar();
subplot(3,3,7)
plot_sdata((P*d_obs-d_obs).data[1], (1f0, 1f0); new_fig=false, name="diff");colorbar();
subplot(3,3,8)
plot_sdata((P*d_obs-d_obs).data[4], (1f0, 1f0); new_fig=false, name="diff");colorbar();
subplot(3,3,9)
plot_sdata((P*d_obs-d_obs).data[nsrc], (1f0, 1f0); new_fig=false, name="diff");colorbar();
tight_layout()
savefig(name*"_data1.png", dpi=300);

figure(figsize=(20,12));
subplot(2,2,1)
plot_sdata(d_obs.data[1], (1f0, 1f0); new_fig=false, name="J*dm");colorbar();
subplot(2,2,2)
plot_sdata((P*d_obs).data[1], (1f0, 1f0); new_fig=false, name="P*J*dm");colorbar();
subplot(2,2,3)
plot_velocity(abs.(fftshift(fft(d_obs.data[1])))/norm(abs.(fftshift(fft(d_obs.data[1]))),Inf), (1f0, 1f0); new_fig=false, name="fft(J*dm)");colorbar();
subplot(2,2,4)
plot_velocity(abs.(fftshift(fft((P*d_obs).data[1])))/norm(abs.(fftshift(fft((P*d_obs).data[1]))),Inf), (1f0, 1f0); new_fig=false, name="fft(P*J*dm)");colorbar();
tight_layout()
savefig(name*"_data2.png", dpi=300);

dm1 = J'*d_obs
dm2 = J'*P'*P*d_obs

figure(figsize=(20,12));
subplot(2,2,1)
plot_simage(dm1.data', d; new_fig=false, name="J'*J*dm", perc=99.5)
subplot(2,2,2)
plot_simage(dm2.data', d; new_fig=false, name="J'*P'*P*J*dm", perc=99.5)
subplot(2,2,3)
plot_velocity(abs.(fftshift(fft(dm1.data')))/norm(abs.(fftshift(fft(dm1.data'))),Inf), d; new_fig=false,cmap="jet",vmax=1, name="fft(J'*J*dm)")
subplot(2,2,4)
plot_velocity(abs.(fftshift(fft(dm2.data')))/norm(abs.(fftshift(fft(dm2.data'))),Inf), d; new_fig=false,cmap="jet",vmax=1, name="fft(J'*P'*P*J*dm)")
tight_layout()
savefig(name*"_1grad.png", dpi=300);

maxit = 5

x1 = 0f0 .* dm
x1,his1 = lsqr!(x1,P*J*Mr,P*d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)

x2 = 0f0 .* dm
x2,his2 = lsqr!(x2,J*Mr,d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)

res1 = his1[:resnorm]
res2 = his2[:resnorm]

u1 = 20*log.(res1./norm(P*d_obs))
u2 = 20*log.(res2./norm(d_obs))

figure(figsize=(20,12));
subplot(1,2,1)
xlabel("iterations")
ylabel("ratio")
PyPlot.plot(res1/norm(P*d_obs),label="P-LSQR")
PyPlot.plot(res2/norm(d_obs),label="LSQR")
title("normalized least squares residual")
legend()
subplot(1,2,2)
xlabel("iterations")
ylabel("dB")
PyPlot.plot(u1,label="P-LSQR")
PyPlot.plot(u2,label="LSQR")
title("normalized log-based least squares residual")
legend()
tight_layout()
savefig(name*"_loss.png", dpi=300);

figure(figsize=(20,12));
subplot(2,2,1)
plot_simage(reshape(Mr*x1,n)', d; new_fig=false, name="P-LSQR", perc=99.5)
subplot(2,2,2)
plot_simage(reshape(Mr*x2,n)', d; new_fig=false, name="LSQR", perc=99.5)
subplot(2,2,3)
plot_velocity(abs.(fftshift(fft(reshape(Mr*x1,n)')))/norm(abs.(fftshift(fft(reshape(Mr*x1,n)'))),Inf), d; new_fig=false,cmap="jet",vmax=1, name="fft(P-LSQR)")
subplot(2,2,4)
plot_velocity(abs.(fftshift(fft(reshape(Mr*x2,n)')))/norm(abs.(fftshift(fft(reshape(Mr*x2,n)'))),Inf), d; new_fig=false,cmap="jet",vmax=1, name="fft(LSQR)")
tight_layout()
savefig(name*"_lsqr.png", dpi=300);
