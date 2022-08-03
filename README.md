# Shampoo.jl

### Overview

[JOLI]:https://github.com/slimgroup/JOLI.jl

This repository includes wave equation preconditioners for operators ``F`` (wave modeling) and ``J`` (born modeling) in the way of correcting their orders. The choices of preconditioners are in ``src/Shampoo.jl``, which contains both data domain (left) preconditioners and model domain (right) preconditioners. To choose a preconditioner, please see the documentation in ``src/Shampoo.jl``.

The preconditioning operators are designed as [JOLI] linear operators. The package dependencies can be found in ``Project.toml``.

### Installation

Please use the command below to install the package

```julia
julia -e 'using Pkg; Pkg.add(url="https://github.com/slimgroup/Shampoo.jl.git")'
```

### Author

If you have any question about the package, please do not hesitate to contact

Ziyi Yin, ziyi.yin@gatech.edu
