# TSRD-derived activity proxies

These six compact CSVs are derived from the 12 user-selected TSRD files and
contain only SPECTRA's generic normalized interface:

```text
band,time_slot,activity
```

They are **not raw TSRD data**.  The preprocessing uses the supplied labels to
create a deterministic activity proxy for software benchmarking.  `band` is
an abstract scheduler channel bucket and must not be interpreted as a physical
frequency mapping. `time_slot` is an abstract record-order time bin.

Mapping used for the supplied files:

- `test_scan`: `config_0.h5` + `config_1.h5`
- `train_scan`: `config_0 (1).h5` + `config_1 (1).h5`
- `val_scan`: `config_0 (2).h5` + `config_1 (2).h5`
- `test_stare`: `config_0 (3).h5` + `config_1 (3).h5`
- `train_stare`: `config_0 (4).h5` + `config_1 (4).h5`
- `val_stare`: `config_0 (5).h5` + `config_1 (5).h5`

The raw HDF5 files are intentionally not bundled in the SPECTRA package.
