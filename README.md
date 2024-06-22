Results
=======

Web version:  http://www.openmathlib.org/BLAS-Benchmarks/

Text timings are visible from the [Actions tab](https://github.com/OpenMathLib/BLAS-Benchmarks/actions), for individual arches --- click on an "ASV Benchmarks" workflow run, then select a desired architecture, and expand the "Run benchmarks" section. Here is [one example](https://github.com/OpenMathLib/BLAS-Benchmarks/actions/runs/9616185161/job/26525115646).


Benchmark suite
===============

The suite is similar to the one which runs on PRs to the OpenBLAS repository
via codspeed: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark/pybench


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

Then either edit `asv.conf.json` to point to the `main` branch:

```diff
$ git diff
diff --git a/asv.conf.json b/asv.conf.json
index ded988b..99e4ff3 100644
--- a/asv.conf.json
+++ b/asv.conf.json
@@ -40,7 +40,7 @@

     // List of branches to benchmark. If not provided, defaults to "main"
     // (for git) or "default" (for mercurial).
-       "branches": ["tracker"], // for git
+       "branches": ["main"], // for git
     // "branches": ["default"],    // for mercurial
```

or update the `tracker` branch to follow `main`:

```
$ git co tracker
$ git revert HEAD
$ git merge --squash main
$ git co main
```

Finally, run

```
$ asv run -v
```
