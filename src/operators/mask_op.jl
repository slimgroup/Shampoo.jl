export mask_op

function mask_op(w::judiWeights{vDT},mask::Array{mDT}) where {vDT,mDT}

    (mDT == vDT) || (mask = reshape(convert(Array{vDT},deepcopy(mask)),size(w.weights[i])))
    w_out = deepcopy(w)
    for i = 1:w.nsrc
        w_out.weights[i] = w.weights[i] .* mask
    end
    return w_out
end

function mask_op(dm::PhysicalParameter{vDT},mask::Array{mDT}) where {vDT,mDT}

    (mDT == vDT) || (mask = reshape(convert(Array{vDT},deepcopy(mask)),dm.n))
    dm_out = deepcopy(dm)
    dm_out.data = dm.data .* mask
    return dm_out

end

function mask_op(dm::Array{vDT},mask::Array{mDT}) where {vDT,mDT}

    (mDT == vDT) || (mask = convert(Array{vDT},deepcopy(mask)))
    dm_out = vec(dm) .* vec(mask)
    return dm_out

end

