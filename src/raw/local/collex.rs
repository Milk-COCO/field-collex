use std::ops::{Deref, DerefMut, Div, Sub};
use num_traits::real::Real;
use span_core::Span;
use crate::raw::local ::map::RawFieldMap;
use crate::raw::local::map::{InsertRawFieldMapError, InsertResult};

pub trait CollexValue<K>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real,
{
    fn pick(&self) -> K;
}


/// RawFieldMap 的高级包装
///
/// Collex 是 Collection Ex 的缩写
///
/// 不需要提供Key，只需要提供Value，Key从Value取得 <br>
/// 使用 [`with_closure`] 来提供闭包以从V得到K <br>
/// 或者 你可以给Value实现`CollexValue<Key>` ，内部会自动使用这个trait中的pick()函数
///
pub struct RawFieldCollex<K,V,F>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real ,
    F: Fn(&V) -> K
{
    map: RawFieldMap<K,V>,
    picker: F,
}

impl<K,V> RawFieldCollex<K,V,fn(&V) -> K>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real,
    V: CollexValue<K>,
{
    pub fn new(span: Span<K>, unit: K) -> Result<Self,(Span<K>,K)> {
        Ok(Self{
            map: RawFieldMap::new(span, unit)?,
            picker: V::pick,
        })
    }
    
    pub fn with_capacity(span: Span<K>, unit: K, capacity: usize) -> Result<Self,(Span<K>,K)> {
        Ok(Self{
            map: RawFieldMap::with_capacity(span, unit, capacity)?,
            picker: V::pick,
        })
    }
}


impl<K,V,F,IE> RawFieldCollex<K,V,F>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize,Error=IE> + Sized + Real,
    F: Fn(&V) -> K
{
    pub fn with_picker(span: Span<K>, unit: K, picker: F) -> Result<Self,(Span<K>,K)> {
        Ok(Self{
            map: RawFieldMap::new(span, unit)?,
            picker,
        })
    }
    
    pub fn with_capacity_picker(span: Span<K>, unit: K, capacity: usize, picker: F) -> Result<Self,(Span<K>,K)> {
        Ok(Self{
            map: RawFieldMap::with_capacity(span, unit, capacity)?,
            picker,
        })
    }
    
    pub fn try_insert(&mut self, value: V) -> super::map::TryInsertResult<V, IE> {
        self.map.try_insert((self.picker)(&value),value)
    }
    
    /// 替换时仅返回V。
    pub fn insert(&mut self, value: V) -> Result<Option<V>, InsertRawFieldMapError<V,IE>>{
        self.map.insert((self.picker)(&value), value).map(|o| o.map(|v| v.1))
    }
    
    /// 替换时保留返回K。
    /// > 但好像没什么用
    pub fn insert_key_value(&mut self, value: V) -> InsertResult<K, V, IE>{
        self.map.insert((self.picker)(&value), value)
    }
}

impl<K,V,F> Deref for RawFieldCollex<K,V,F>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real ,
    F: Fn(&V) -> K
{
    type Target = RawFieldMap<K,V>;
    
    /// 只有一些方法并需要进行上层包装。比如插入相关。
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<K,V,F> DerefMut for RawFieldCollex<K,V,F>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real ,
    F: Fn(&V) -> K
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}
