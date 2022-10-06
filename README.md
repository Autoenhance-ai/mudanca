# Pyshift

A perspective correction library for OpenCV images ported from the Darktable Ashift module (https://github.com/darktable-org/darktable/blob/release-4.1.0/src/iop/ashift.c)
which itself was based on ShiftN (https://www.shiftn.de).

It has been adapted to consume the results of Open CV's built-in LSD algorithmn.

## Roadmap

- [] Fix and re-enable removal of outliers for better results
- [] Remove preprocessing steps (i.e Sharpening) or allow them to be turned off / on 
- [] Expose flags to allow for more customization of algoritmn (i.e Disable/Enable Rotation)
- [] Eventually port and open source as Open CV contrib module
- [] Remove and disable debugging code
- [] Validate inputsd