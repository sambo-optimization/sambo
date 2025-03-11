SAMBO Optimization Benchmark
============================
This package compares [SAMBO](https://sambo-optimization.github.io)
to popular optimization `methods.py` by running the algorithms on 
test objective `funcs.py`.

Run in project root:
```shell
python -m benchmark |
    tee benchmark-$(date --iso-8601=minutes).txt
```

See [published result](https://sambo-optimization.github.io/#benchmark).
