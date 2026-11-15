# P3 short contact-event correction

[Unchanged physical prediction XY](../handoff_loss/README.md) · [All72 event errors](event_errors.png) · [5ms sampling example](sampling_example.png)

Native5ms events now produce cumulative20ms labels: the brief event remains visible after it occurs. No trajectory interpolation, new physics, or relabeling of task success. Readouts use only frozen predicted physical state, model memory and elapsed skill time; actual event times appear only in supervision/evaluation.

All3readout seeds report both events for all72DEV cases. Selected readout seed1/update600 has no-wheel-contact MAE10.83ms and far-contact MAE12.57ms. The same20ms-grid TRAIN-median control is10.14ms/7.01ms; its continuous-time version is6.11ms/6.04ms. **No learned-event advantage over fixed timing is established.** The control is a diagnostic baseline, never a replacement for the actual robot handoff logic.

The physical model remains seed0/update2275. A real-checkpoint GPU test confirms bitwise100=40+serialize+60 for both physical prediction and cumulative event memory, and zero physical change with the readout attached. Three unit tests cover short-event preservation, event continuation, and combined serialized continuation. Negative cases and earlier sampling failures remain in previous reports.

These are1–2native no-wheel-contact samples per trajectory, not proof of sustained aerial flight. Model development remains limited to the declared gap and known topology. Old frozen M1 margins reject all viable continuation cases; full-chain calibration and planning remain unfinished.
