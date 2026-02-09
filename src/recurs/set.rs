use num_traits::real::Real;
use span_core::Span;
use crate::raw::set;
use crate::raw::set::RawFieldSet;

/// 上层包装。每个块可以存多个内容（通过递归结构实现）
pub struct FieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    raw: RawFieldSet<V>
}

impl<V> FieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    
    /// 提供span与unit，构建一个RawFieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<V>, unit: V) -> set::NewResult<Self,V> {
        Ok(Self {
            raw: RawFieldSet::new(span, unit)?
        })
    }
    
}