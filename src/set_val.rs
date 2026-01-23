/// Zero-Sized Type (ZST) for internal `FieldSet` values.
/// Used instead of `()` to differentiate between:
/// * `FieldMap<T, ()>` (possible user-defined map)
/// * `FieldMap<T, SetValZST>` (internal set representation)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Default)]
pub(crate) struct SetValZST;

/// A trait to differentiate between `FieldMap` and `FieldSet` values.
/// Returns `true` only for type `SetValZST`, `false` for all other types (blanket implementation).
/// `TypeId` requires a `'static` lifetime, use of this trait avoids that restriction.
///
/// [`TypeId`]: core::any::TypeId
pub(crate) trait IsSetVal {
    fn is_set_val() -> bool;
}

// Blanket implementation
impl<V> IsSetVal for V {
    default fn is_set_val() -> bool {
        false
    }
}

// Specialization
impl IsSetVal for SetValZST {
    fn is_set_val() -> bool {
        true
    }
}
