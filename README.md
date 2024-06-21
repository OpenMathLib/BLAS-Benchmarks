Results
=======

http://www.openmathlib.org/BLAS-Benchmarks/

CI orchestration
================

We run the benchmark suite on a cron schedule.

We use three branches:

  - `main` : contains benchmark code, the BLAS/LAPACK wrappers etc. This is the branch
    you want to merge feature branches to;
  - `gh-pages` : contains the html site + the `results/` from past runs; Commits
    list the OpenBLAS wheel versions and configurations. This branch is auto-updated on
    each CI run from main;
  - `tracker` : `asv` benchmarks this branch. Mirrors `main`.
    Is auto-updated on each CI run.

The reason for this setup is that on each run,  `asv`

  - checks out the branch it benchmarks,
  - builds the project, and
  - labels the results by the top commit in that branch

Since we use the `scipy_openblas32` nightly builds, our main branch itself does
not track changes in OpenBLAS (the wheels do).
Were we to just benchmark `main`, all our runs would have been attributed
to the same commit --- effectively, a new CI run would overwrite the previous one.

To sidestep this, on each CI run we point `asv` to the `tracker` branch, make it
functionally equivalent to `main`, and add a new commit at the beginning of a CI
benchmarking run.


Running benchmarks locally
==========================


To run the benchmark suite locally, first do

```
$ python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > scipy_openblas.pc
$ export PKG_CONFIG_PATH=$PWD
```

and then

```
$ asv run -v
```
