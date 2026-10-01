# Dataset folder

Not needed on Kaggle or when the automatic download is used (see the main README).

For a local copy, place the SmartPNT-POS folders here (any depth), for example:

```
Dataset/
├── Data01_20230102_ISA-100C_Vehicle_Complex/
├── Data02_20220309_ISA-100C_Vehicle_Complex/
├── LEO_TLE/           # TLE files (*.txt) of the LEO satellites around both days
└── products/          # *.sp3, *.clk, brdm* of both days, if not inside the data folders
```

The products are searched in the data folder, then in `products/`, then in `Dataset/` itself (on Kaggle they are at
the root of the dataset). An SP3 or CLK file that does not cover the dataset stops the run with an error.

Source: https://www.kaggle.com/datasets/fengzhusgg/smartpnt-pos (not redistributed here).
