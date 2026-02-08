use num_traits::real::Real;
use crate::raw::set::RawFieldSet;

/// 上层包装。消除限制
pub struct FieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    raw: RawFieldSet<V>
}

