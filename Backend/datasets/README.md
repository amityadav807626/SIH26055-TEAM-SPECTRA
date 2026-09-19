# Dataset adapters

`activity_adapter.py` accepts only a neutral normalized CSV schema:

```text
band,time_slot,activity
0,0,1
2,1,1
2,2,0
```

It validates the records and converts them into the same `(time_slot, band)` activity
representation used by the software simulator. It intentionally contains no
source-specific radar/emitter parsing.
