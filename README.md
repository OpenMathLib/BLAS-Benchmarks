To run the benchmark suite locally, first do

```
$ python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > openblas.pc
$ export PKG_CONFIG_PATH=$PWD
```

and then

```
$ asv run -v
```
