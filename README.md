# Shampoo.jl

### Overview

[JOLI]:https://github.com/slimgroup/JOLI.jl

This repository includes wave equation preconditioners for operators ``F`` (wave modeling) and ``J`` (born modeling) in the way of correcting their orders. The choices of preconditioners are in ``src/Shampoo.jl``. Currently, we have a data-domain preconditioner for fractional integration/derivative. The default one is half integration (order is 0.5).

The preconditioning operators are designed as [JOLI] linear operators. The package dependencies can be found in ``Project.toml``.

### Installation

Please use the command below to install the package

```julia
julia -e 'using Pkg; Pkg.add(url="https://github.com/slimgroup/Shampoo.jl.git")'
```

### Author

Ziyi Yin, ziyi.yin@gatech.edu

We welcome issues and pull requests.