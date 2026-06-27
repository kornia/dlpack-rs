//! Portable DLPack bindings: raw FFI ([`ffi`]), safe builders ([`safe`]),
//! and optional pyo3 capsule glue (feature `pyo3`).
pub mod ffi;
#[cfg(feature = "pyo3")]
pub mod pyo3_glue;
pub mod safe;
