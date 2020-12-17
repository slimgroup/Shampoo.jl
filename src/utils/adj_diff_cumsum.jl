# adjoint of diff/cumsum operators for judiVector

export adjoint_diff, adjoint_cumsum

function adjoint_diff(d_obs::judiVector)
    d_out = deepcopy(d_obs)
    for i = 1:d_obs.nsrc
        d_out.data[i][1:end-1,:] = d_obs.data[i][1:end-1,:] - d_obs.data[i][2:end,:]
    end
    return d_out
end

function adjoint_cumsum(d_obs::judiVector)
    d_out = deepcopy(d_obs)
    for i = 1:d_obs.nsrc
        d_out.data[i] = reverse(cumsum(reverse(d_obs,dims=1),dims=1),dims=1)
    end
    return d_out
end
