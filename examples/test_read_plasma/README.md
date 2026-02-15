# test_read_plasma

This folder validates `compute plasma/fields` in two modes and plots `log10(ne)` in `(R,Z)` with core masking from `core.txt`.

## Active files

- `in.file`: file-based plasma/bfield input
- `in.constant`: constant plasma-field input
- `plasma.h5`: plasma source data for file mode
- `bfield.h5`: magnetic source data for file mode
- `core.txt`: closed core polygon used to mask empty core in plots
- `test_neutral.py`: run + read + plot driver
- `output/`: generated dumps, CSV, and PNG

## Run file mode

```bash
/usr/bin/python3 test_neutral.py --case file --np 4
```

Outputs:
- `output/plasma_grid.ne`
- `output/ne_rz_file.csv`
- `output/ne_rz_file.png`

## Run constant mode

```bash
/usr/bin/python3 test_neutral.py --case constant --np 4
```

Outputs:
- `output/plasma_grid.constant.ne`
- `output/ne_rz_constant.csv`
- `output/ne_rz_constant.png`
