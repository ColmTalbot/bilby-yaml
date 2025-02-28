A command-line interface to run `bilby` using a YAML configuration file.

> [!NOTE]
>
> Usage: `bilby <config.yaml>`

This executable will load the prior and likelihood from the provided YAML file
and run the sampler. All arguments in the top level of the configuration file
will be passed to `bilby.core.sampler.run_sampler` except for
`non-sampled` and `post`. The former is ignored during the running
and can be used to store anchors and templates. The latter is a dictionary
specifying methods of the result object to call after the sampler has run,
e.g., plotting routines.

The following script is slightly modfied from the multivariate Gaussian example in
the bilby repository. Important aspects are:

- The use of the `!CallWithKwargs` and `!CallWithoutKwargs` tags to
  load classes/functions with and without keyword arguments, respectively.
- The `!Seed` tag sets the global seed in `bilby`.
- The `non-sampled` section is used to define the multivariate distribution
  that is reused in the prior.
- A corner plot is requested using `post`. This passes no additional arguments.

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