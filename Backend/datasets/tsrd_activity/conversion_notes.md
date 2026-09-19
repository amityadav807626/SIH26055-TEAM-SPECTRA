# TSRD conversion / validation notes

The 12 user-selected HDF5 files were converted outside the SPECTRA runtime to CSV for inspection. All 12 CSVs were read successfully and contained the expected six columns: `ToA`, `Frequency`, `PulseWidth`, `AoA`, `Amplitude`, `label`.

For the SPECTRA package, only six compact normalized activity proxies are bundled. Each proxy combines the two selected files for one split and contains exactly 260 time slots × 30 abstract scheduler channels. Activity is derived from the supplied classification labels (`label != 0`); channel buckets and time bins are abstract software-test indices, not physical frequency mappings.

The raw HDF5/PDW data and the large intermediate CSVs are intentionally excluded from the final package.
