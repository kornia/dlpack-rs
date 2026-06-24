//! Re-export of auto-generated DLPack FFI bindings from `dlpack-sys`.
//! The bindgen-generated types replace the former hand-written structs.
#[allow(unused_imports)]
pub use dlpack_sys::*;

// ── Stable public aliases ──────────────────────────────────────────────────────
// These insulate consumers from bindgen naming (DLDeviceType::kDLCPU, etc.) and
// match the struct field types so comparisons compile without a cast at call sites.
//
// DLDevice::device_type  → DLDeviceType::Type = c_uint = u32
// DLDataType::code       → u8
// DLManagedTensorVersioned::flags → u64

/// CPU device type constant — directly comparable to `DLDevice::device_type` (u32).
pub const K_DL_CPU: u32 = DLDeviceType::kDLCPU;

/// CUDA GPU device type constant — directly comparable to `DLDevice::device_type` (u32).
pub const K_DL_CUDA: u32 = DLDeviceType::kDLCUDA;

/// Unsigned-integer dtype code — directly comparable to `DLDataType::code` (u8).
pub const K_DL_UINT: u8 = DLDataTypeCode::kDLUInt as u8;

/// Signed-integer dtype code — directly comparable to `DLDataType::code` (u8).
pub const K_DL_INT: u8 = DLDataTypeCode::kDLInt as u8;

/// IEEE floating-point dtype code — directly comparable to `DLDataType::code` (u8).
pub const K_DL_FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;

/// Boolean dtype code — directly comparable to `DLDataType::code` (u8).
pub const K_DL_BOOL: u8 = DLDataTypeCode::kDLBool as u8;

/// Read-only bitmask for `DLManagedTensorVersioned::flags` (u64).
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = dlpack_sys::DLPACK_FLAG_BITMASK_READ_ONLY as u64;

/// Is-copied bitmask for `DLManagedTensorVersioned::flags` (u64).
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = dlpack_sys::DLPACK_FLAG_BITMASK_IS_COPIED as u64;
