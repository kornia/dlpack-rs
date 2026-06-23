//! Raw `#[repr(C)]` DLPack ABI types. No dependencies; portable across arches.
use core::ffi::c_void;

// DLDeviceType (C enum = int). Only the common kinds; extend as needed.
pub const K_DL_CPU: i32 = 1;
pub const K_DL_CUDA: i32 = 2;
pub const K_DL_CUDA_HOST: i32 = 3;
pub const K_DL_OPENCL: i32 = 4;
pub const K_DL_VULKAN: i32 = 7;
pub const K_DL_METAL: i32 = 8;
pub const K_DL_ROCM: i32 = 10;
pub const K_DL_CUDA_MANAGED: i32 = 13;
pub const K_DL_ONE_API: i32 = 14;

// DLDataTypeCode
pub const K_DL_INT: u8 = 0;
pub const K_DL_UINT: u8 = 1;
pub const K_DL_FLOAT: u8 = 2;
pub const K_DL_OPAQUE_HANDLE: u8 = 3;
pub const K_DL_BFLOAT: u8 = 4;
pub const K_DL_COMPLEX: u8 = 5;
pub const K_DL_BOOL: u8 = 6;

// DLPack 1.0 flags
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64, // null => compact row-major
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};
    #[test]
    fn abi_layout() {
        assert_eq!(size_of::<DLDevice>(), 8);
        assert_eq!(size_of::<DLDataType>(), 4);
        // DLTensor layout on LP64 (aarch64 measured):
        //   data(8) + DLDevice(8) + ndim(4) + DLDataType(4) + shape(8) + strides(8) + byte_offset(8) = 48
        // ndim and dtype pack contiguously (dtype align=2, so no gap before ndim+dtype block);
        // shape is 8-aligned and falls at offset 24 (8+8+4+4=24), no extra padding needed.
        // NOTE: the brief suggested 56 (with two 4-byte pads), but the actual aarch64 layout
        // is 48. We lock the correct measured value here.
        assert_eq!(size_of::<DLTensor>(), 48);
        assert_eq!(align_of::<DLTensor>(), 8);
        // DLManagedTensor: DLTensor(48) + manager_ctx(8) + deleter fn-ptr(8) = 64
        assert_eq!(size_of::<DLManagedTensor>(), 64);
        assert_eq!(size_of::<DLPackVersion>(), 8);
        // DLManagedTensorVersioned on LP64 (aarch64 measured):
        //   DLPackVersion(8) + manager_ctx ptr(8) + deleter fn-ptr(8) + flags u64(8) + DLTensor(48) = 80
        assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
        assert_eq!(K_DL_CPU, 1);
        assert_eq!(K_DL_FLOAT, 2);
    }
}
