"""A command-line interface to run :code:`bilby` using a YAML configuration file.

Usage: bilby <config.yaml>

This executable will load the prior and likelihood from the provided YAML file
and run the sampler. All arguments in the top level of the configuration file
will be passed to :func:`bilby.core.sampler.run_sampler` except for
:code:`non-sampled` and :code:`post`. The former is ignored during the running
and can be used to store anchors and templates. The latter is a dictionary
specifying methods of the result object to call after the sampler has run,
e.g., plotting routines.

The following script is slightly modfied from the multivariate Gaussian example in
the bilby repository. Important aspects are:

- The use of the :code:`!CallWithKwargs` and :code:`!CallWithoutKwargs` tags to
  load classes/functions with and without keyword arguments, respectively.
- The :code:`!Seed` tag sets the global seed in :code:`bilby`.
- The :code:`non-sampled` section is used to define the multivariate distribution
  that is reused in the prior.
- A corner plot is requested using :code:`post`. This passes no additional arguments.

```yaml
label: multivariate_gaussian
outdir: outdir
seed: !Seed 123
sampler: dynesty
nlive: 10000
clean: true
resume: false
non-sampled:
  mvg: !CallWithKwargs:bilby.core.prior.MultivariateGaussianDist &mvg
    names: ["m", "c"]
    nmodes: 2
    mus: [[-5.0, -5.0], [5.0, 5.0]]
    corrcoefs: [[[1.0, -0.7], [-0.7, 1.0]], [[1.0, 0.7], [0.7, 1.0]]]
    sigmas: [[1.5, 1.5], [2.1, 2.1]]
    weights: [1.0, 3.0]
priors: !CallWithoutKwargs:bilby.core.prior.PriorDict
  m: !CallWithKwargs:bilby.core.prior.MultivariateGaussian
    dist: *mvg
    name: m
    latex_label: $m$
  c: !CallWithKwargs:bilby.core.prior.MultivariateGaussian
    dist: *mvg
    name: c
    latex_label: $c$
likelihood: !CallWithKwargs:bilby.core.likelihood.GaussianLikelihood
  func: !Executable __main__.model
  x: 0.0
  y: 0.0
  sigma: 3000.0
post:
  plot_corner:
```
"""

import os
import sys
from functools import partial
from importlib import import_module

import bilby
import yaml


def get_method(loader, node):
    """
    Determine the appropriate load method to use given the node type.
    """
    match type(node):
        case yaml.nodes.ScalarNode:
            method = loader.construct_scalar
        case yaml.nodes.SequenceNode:
            method = loader.construct_sequence
        case yaml.nodes.MappingNode:
            method = loader.construct_mapping
        case _:
            raise ValueError(f"Unknown kind {type(node)} for loader")
    return method


def generic_load(func, *, multi=False, **kwargs):
    """
    A wrapper function to load and input using the provided function.

    Parameters
    ----------
    func: callable
        The function to use to load the input, this should have signature
        :code:`func(input: [dict, sequence, str], **kwargs) -> obj`.
    multi: bool, optional
        Whether the function takes a :code:`tag_suffix` argument as provided
        by :code:`multi_constructor` methods.
    kwargs: dict
        Additional keyword arguments to pass to the loader, e.g., :code:`deep`
        for recursively loading mappings.
    """    
    if multi:
        def wrapper(loader, tag_suffix, node):
            method = get_method(loader, node)
            tag_suffix = tag_suffix.strip(":")
            return func(method(node, **kwargs), tag_suffix)
    else:
        def wrapper(loader, node):
            method = get_method(loader, node)
            return func(method(node, **kwargs))

    return wrapper


def load_numpy(conf):
    """Load a sequence into a numpy array."""
    import numpy as np
    return np.array(conf)


def load_executable(conf):
    """
    Load a class or function from a module, e.g.,
    :code:`bilby.core.likelihood.GaussianLikelihood`.

    Parameters
    ----------
    conf: str
        The module path and class or function
    
    Returns
    -------
    obj
        The loaded class or function
    """
    module, path = conf.rsplit(".", maxsplit=1)
    return getattr(import_module(module), path)


def _extract_kwargs(conf):
    if not isinstance(conf, dict):
        return dict()
    magic_keys = ["__method__", "__path__"]
    return {key: val for key, val in conf.items() if key not in magic_keys}


def load_class_with_kwargs(conf, tag_suffix=None, **kwargs):
    """
    Load and instantiate a class that takes a dictionary of keyword arguments,
    e.g., :code:`bilby.core.prior.Prior`.
    """
    if tag_suffix is None:
        tag_suffix = conf["__path__"]
    cls_ = load_executable(tag_suffix)
    kwargs.update(_extract_kwargs(conf))
    if "__method__" in conf:
        meth = getattr(cls_, conf["__method__"])
    else:
        meth = cls_
    return meth(**kwargs)


def load_class_without_kwargs(conf, tag_suffix=None):
    """
    Load and instantiate a class that takes a single input argument, e.g., a
    dictionary as the only argument for :func:`bilby.core.prior.PriorDict`.
    """
    if tag_suffix is None:
        tag_suffix = conf["__path__"]
    cls_ = load_executable(tag_suffix)
    return cls_(_extract_kwargs(conf))


def load_seed(loader, node):
    """
    Load the provided seed and set the global seed in :code:`bilby`.
    """
    rseed = int(loader.construct_scalar(node))
    bilby.core.utils.random.seed(rseed)
    return rseed


class BilbyLoader(yaml.SafeLoader):
    pass


BilbyLoader.add_constructor(
    "!CallWithoutKwargs", generic_load(load_class_without_kwargs)
)
BilbyLoader.add_multi_constructor(
    "!CallWithoutKwargs", generic_load(load_class_without_kwargs, multi=True)
)
BilbyLoader.add_constructor("!Executable", generic_load(load_executable))
BilbyLoader.add_constructor(
    "!CallWithKwargs", generic_load(load_class_with_kwargs, deep=True)
)
BilbyLoader.add_multi_constructor(
    "!CallWithKwargs", generic_load(load_class_with_kwargs, multi=True, deep=True)
)
BilbyLoader.add_constructor("!numpy.ndarray", generic_load(load_numpy))
BilbyLoader.add_constructor("!Seed", load_seed)


class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, N, injection_parameters):
        super().__init__(dict())
        self.N = N
        self.injection_parameters = injection_parameters

    def log_likelihood(self):
        import numpy as np
        mu = self.parameters["mu"]
        sigma = self.parameters["sigma"]
        true_mu = self.injection_parameters["mu"]
        true_sigma = self.injection_parameters["sigma"]
        dkl = np.log(sigma / true_sigma) + (true_sigma**2 + (mu - true_mu) ** 2) / (2 * sigma**2) - 0.5
        return -self.N * dkl


def linear_model(x, m, c):
    return m * x + c


def main():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        fname = sys.argv[1]
    else:
        print(__doc__)
        sys.exit()

    with open(fname, "r") as ff:
        config = yaml.load(ff, Loader=BilbyLoader)

    config.pop("non-sampled", None)
    post = config.pop("post", list())

    result = bilby.run_sampler(**config)

    for func, kwargs in post.items():
        if kwargs is None:
            kwargs = dict()
        meth = getattr(result, func, None)
        if meth is None:
            raise ValueError(f"Function {func} not found in result")
        meth(**kwargs)


if __name__ == "__main__":
    main()
