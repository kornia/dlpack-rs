# Vendored DLPack header

Source: https://github.com/dmlc/dlpack (the canonical "original repo")
File:   include/dlpack/dlpack.h
Version: v1.3
Blob SHA (git hash-object): 5836c7e9f97d593b63131b7931625416efb1ad71

Vendored in-tree (NOT a git submodule) so it ships in the crates.io artifact AND
in cargo git dependencies (cargo fetches neither submodules nor network at build).
`dlpack-sys/build.rs` runs bindgen against this header into OUT_DIR per target arch
(so aarch64/x86 both work; no committed per-arch dump).

## Updating to a newer DLPack
    gh api repos/dmlc/dlpack/contents/include/dlpack/dlpack.h --jq .content \
      | base64 -d > dlpack-sys/vendor/dlpack/dlpack.h
then bump Version/SHA above and re-run the dlpack-sys layout tests.
