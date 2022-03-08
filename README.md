# Neural networks for Helmholtz--Hodge decomposition

In this project, we investigate physics-informed neural networks (PINNs) that
are constructed in such way that they separate potential (curl-free) and
solenoidal (divergence-free) vector fields, what is known as Helmholtz--Hodge
decomposition.

## Setup

1. Clone the project with `git` via SSH:

       git clone git@github.com:dmitry-kabanov/hhpinn.git

    or HTTPS:

       git clone https://github.com/dmitry-kabanov/hhpinn.git

2. Install [conda](https://docs.conda.io) environment for the project:

       conda env create -f environment.yml [-n env-name]

where optional argument `env-name` has the default value `hhpinn`.
