# Mudanca

A perspective correction library for OpenCV images ported from the Darktable Ashift module (https://github.com/darktable-org/darktable/blob/release-4.1.0/src/iop/ashift.c)
which itself was based on ShiftN (https://www.shiftn.de).

We have only ported the correction code to allow you to provide your own line detection code.

## Installing

```
pip install mudanca
```

## Running Tests

```
pip install . && python -m pytest
```