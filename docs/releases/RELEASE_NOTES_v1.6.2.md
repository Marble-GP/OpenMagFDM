# OpenMagFDM v1.6.2 release notes

Status: stable release.

## WebUI fix

Loading an image into the Run & Preview Input Image viewer could cause the
preview frame to grow repeatedly. The ruler canvases, zoom stage and image
were participating in an unconstrained intrinsic-size feedback loop: updating
the canvas backing dimensions changed the grid's minimum content size, which
changed the viewport again.

v1.6.2 fixes this by:

- bounding the preview viewport to the panel width and a 120–400 px height;
- using zero-minimum grid tracks so intrinsic image/canvas dimensions cannot
  enlarge the parent; and
- constraining the zoom stage and image to the viewport while preserving
  wheel zoom, middle-button pan, reset and pixel rulers.

No solver, project-file or analysis-format behavior was changed.

## Verification

- `node --check webui/public/app.js` passes.
- A Chromium browser test loaded a 4000×2000 image and observed a stable
  548×420 px ruler wrapper and 522×400 px image viewport over repeated layout
  samples.
- The existing WebUI lifecycle suite remains unchanged; its local sandbox run
  had one unrelated Windows `EPERM` failure while creating a test output
  directory.
