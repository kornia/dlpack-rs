#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;
    #[test]
    fn abi_layout() {
        assert_eq!(size_of::<DLTensor>(), 48);
        assert_eq!(size_of::<DLManagedTensor>(), 64);
        assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
    }
}
