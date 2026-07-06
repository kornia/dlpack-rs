#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
include!("bindings.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn abi_layout_lp64() {
        assert_eq!(size_of::<DLTensor>(), 48);
        assert_eq!(size_of::<DLManagedTensor>(), 64);
        assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
    }
}
