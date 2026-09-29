# Vendored 3Dmol.js

`streamlit_app.py` inlines `3Dmol-min.js` into the structure viewer's iframe
so the user's browser never loads JavaScript from an external host.

- Library: 3Dmol.js 2.5.5 (<https://3dmol.org>, <https://github.com/3dmol/3Dmol.js>)
- File: `3Dmol-min.js`, downloaded unmodified from
  `https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.5/3Dmol-min.js`
  (537792 bytes)
- sha256: `f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016`
- sha512 (base64): `rk2gI8FYzSbiZnDZ9M70SEhemyYIDwQQhb1zH9eK8kglMsp7bKJdu5+akb+wlTQxC9DiMmoMTNUQ0Z0Q/trdyw==`
  -- identical to the SRI hash cdnjs publishes for this file.
- License: BSD-3-Clause (`LICENSE`, from the 3Dmol.js repository; it also
  lists the licenses of bundled GLmol / Three.js / jQuery code), plus the
  build's own notice `3Dmol-min.js.LICENSE.txt` (from the npm package
  `3dmol@2.5.5`).

The library contains helper URLs (RCSB, PubChem) for fetching structures
by ID; the app never calls them -- it only passes Chai's own CIF text to
`viewer.addModel`.

To upgrade: replace `3Dmol-min.js`, update the version/hashes above
(`tests/test_offline.py` checks the sha256), and re-check the result page.
