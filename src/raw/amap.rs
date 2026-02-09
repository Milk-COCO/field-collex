use std::hash::Hash;
use std::ops::{Div, Mul, Sub};
use num_traits::real::Real;
use span_core::Span;
use super::{set,map};
use super::map::RawFieldMap;
pub use super::map::InsertRawFieldMapError;

pub trait AMapValue<K>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + Into<usize> + Sized + Real,
{
    fn pick(&self) -> K;
}


/// RawFieldMap 的高级包装
///
/// AMap 意思是 Auto Map
///
/// 不需要提供Key，只需要提供Value，Key从Value取得，这就是auto的意味 <br>
/// 使用 [`with_closure`] 来提供闭包以从V得到K <br>
/// 或者 你可以给Value实现`AMapValue<Key>` ，内部会自动使用这个trait中的pick()函数
///
pub struct RawFieldAMap<K,V,F>
where
    K: Ord + Real + Into<usize>,
    K: Hash,
    F: Fn(&V) -> K
{
    map: RawFieldMap<K,V>,
    picker: F,
}

impl<K,V> RawFieldAMap<K,V,fn(&V) -> K>
where
    K: Ord + Real + Into<usize>,
    K: Hash,
    V: AMapValue<K>,
{
    pub fn new(span: Span<K>, unit: K) -> set::NewResult<Self, K> {
        Ok(Self{
            map: RawFieldMap::new(span, unit)?,
            picker: V::pick,
        })
    }
    
    pub fn with_capacity(span: Span<K>, unit: K, capacity: usize) -> set::WithCapacityResult<Self, K> {
        Ok(Self{
            map: RawFieldMap::with_capacity(span, unit, capacity)?,
            picker: V::pick,
        })
    }
}


impl<K,V,F> RawFieldAMap<K,V,F>
where
    K: Ord + Real + Into<usize>,
    K: Hash,
    F: Fn(&V) -> K
{
    pub fn with_picker(span: Span<K>, unit: K, picker: F) -> set::NewResult<Self, K> {
        Ok(Self {
            map: RawFieldMap::new(span, unit)?,
            picker,
        })
    }
    
    pub fn with_capacity_picker(span: Span<K>, unit: K, capacity: usize, picker: F) -> set::WithCapacityResult<Self, K> {
        Ok(Self {
            map: RawFieldMap::with_capacity(span, unit, capacity)?,
            picker,
        })
    }
    
    pub fn span(&self) -> &Span<K> {
        &self.map.span()
    }
    
    pub fn unit(&self) -> &K {
        &self.map.unit()
    }
    
    /// 返回最大块数量
    ///
    /// 若Span是无限区间，返回Ok(None) <br>
    pub fn size(&self) -> Option<usize> {
        self.map.size()
    }
    
    /// 返回已存在的块的数量
    ///
    /// 此值应小于等于最大块数量
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    
    /// Returns the total number of elements the inner vector can hold without reallocating.
    ///
    /// 此值应小于等于最大块数量
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }
    
    /// 判断对应块是非空
    pub fn is_thing(&self, idx: usize) -> bool {
        self.map.is_thing(idx)
    }
    
    /// 通过索引返回 块的值 的引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub fn thing(&self, idx: usize) -> Option<&V> {
        self.map.thing(idx)
    }
    
    /// 通过索引返回 块的值 的可变引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub fn thing_mut(&mut self, idx: usize) -> Option<&mut V> {
        self.map.thing_mut(idx)
    }
    
    
    /// 计算指定值对应的块索引
    ///
    /// 此无任何前置检查，只会机械地返回目标相对于初始位置（区间的左端点）可能处于第几个块，但不确保这个块是否合法。<br>
    /// 包含前置检查的版本是[`get_index`]
    #[inline(always)]
    pub fn idx_of(&self, key: K) -> usize {
        self.map.idx_of(key)
    }
    
    /// 查找对应键是否存在
    ///
    pub fn contains_key(&self, value: K) -> bool {
        self.map.contains_key(value)
    }
    
    /// 查找是否已存在与提供目标的Key相同的Value
    ///
    pub fn contains(&self, value: &V) -> bool {
        self.map.contains_key((self.picker)(&value))
    }
    
    
    /// 通过索引得到当前或上一个非空块的(索引,值引用)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，或索引越界，返回None <br>
    pub fn get_prev(&self,idx: usize) -> Option<(usize, &V)> {
        self.map.get_prev(idx).map(|t| (t.0,t.2))
    }
    
    /// 通过索引得到当前或下一个非空块的(索引,值引用)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，或索引越界，返回None <br>
    pub fn get_next(&self,idx: usize) -> Option<(usize, &V)> {
        self.map.get_next(idx).map(|t| (t.0,t.2))
    }
    
    
    /// 通过索引得到当前或上一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，或索引越界，返回None <br>
    pub fn get_prev_index(&self,idx: usize) -> Option<usize> {
        self.map.get_prev_index(idx)
    }
    
    /// 通过索引得到当前或下一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，或索引越界，返回None <br>
    pub fn get_next_index(&self,idx: usize) -> Option<usize> {
        self.map.get_next_index(idx)
    }
    
    
    pub fn try_insert(&mut self, value: V) -> map::TryInsertResult<V> {
        self.map.try_insert((self.picker)(&value), value)
    }
    
    /// 插入或替换值
    ///
    /// 若对应块已有值，新值将替换原值，返回Ok(Some(V))包裹原值。<br>
    /// 若无值，插入新值返回 Ok(None)。
    ///
    pub fn insert(&mut self, value: V) -> Result<Option<V>, InsertRawFieldMapError<V>> {
        self.map.insert((self.picker)(&value), value).map(|o| o.map(|v| v.1))
    }
    
    /// 用索引指定替换块的键
    ///
    /// 成功则返回其原键
    pub fn replace_key_index(&mut self, idx: usize, key: K) -> set::ReplaceIndexResult<K>
    where
        K: Mul<usize, Output = K>,
    {
        self.map.replace_key_index(idx, key)
    }
    
    /// 用索引指定替换块的键，但无法替换时panic
    ///
    /// 返回其原键
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    pub fn unchecked_replace_key_index(&mut self, idx: usize, key: K) -> K
    where
        K: Mul<usize, Output = K> + std::fmt::Debug,
    {
        self.map.unchecked_replace_key_index(idx, key)
    }
    
    
    /// 用索引指定替换块的值
    ///
    /// 成功则返回其原值
    pub fn replace_index(&mut self, idx: usize, value: V) -> map::ReplaceIndexResult<V> {
        self.map.replace_index(idx, value)
    }
    
    /// 用索引指定替换块的值，但无法替换时panic
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    pub fn unchecked_replace_index(&mut self, idx: usize, value: V) -> V {
        self.map.unchecked_replace_index(idx, value)
    }
    
    /// 用索引指定清空块。
    ///
    /// 若指定块非空，返回(键,值)。
    pub fn remove_index(&mut self, idx: usize) -> set::RemoveIndexResult<V> {
        self.map.remove_index(idx).map(|t| t.1)
    }
    
    /// 用索引指定清空块，但无法清空时panic
    ///
    /// 返回原(键,值)。
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    pub fn unchecked_remove_index(&mut self, idx: usize) -> V {
        self.map.unchecked_remove_index(idx).1
    }
    
    /// 用键清空对应块。
    ///
    /// 若指定块非空，返回原(键,值)。
    pub fn remove(&mut self, key: K) -> set::RemoveResult<V> {
        self.map.remove(key).map(|t| t.1)
    }
    
    /// 用键清空对应块，但无法清空时panic
    ///
    /// 返回原(键,值)。
    ///
    /// # Panics
    /// 同[`unchecked_get_index`] + [`unchecked_remove_index`]
    pub fn unchecked_remove(&mut self, key: K) -> V {
        self.map.unchecked_remove(key).1
    }
    
    
    
    /// 找到最近的小于等于 target 的键，返回对应值的引用
    ///
    pub fn find_le(&self, target: K) -> set::FindResult<&V> {
        self.map.find_le(target)
    }
    
    /// 找到最近的小于 target 的键，返回对应值的引用
    ///
    pub fn find_lt(&self, target: K) -> set::FindResult<&V> {
        self.map.find_lt(target)
    }
    
    /// 找到最近的大于等于 target 的键，返回对应值的引用
    ///
    pub fn find_ge(&self, target: K) -> set::FindResult<&V> {
        self.map.find_ge(target)
    }
    
    /// 找到最近的大于 target 的键，返回对应值的引用
    ///
    pub fn find_gt(&self, target: K) -> set::FindResult<&V> {
        self.map.find_gt(target)
    }
    
    /// 找到最近的小于等于 target 的键，返回索引
    ///
    pub fn find_index_le(&self, target: K) -> set::FindResult<usize> {
        self.map.find_index_le(target)
    }
    
    /// 找到最近的小于 target 的键，返回索引
    ///
    pub fn find_index_lt(&self, target: K) -> set::FindResult<usize> {
        self.map.find_index_lt(target)
    }
    
    /// 找到最近的大于等于 target 的键，返回索引
    ///
    pub fn find_index_ge(&self, target: K) -> set::FindResult<usize> {
        self.map.find_index_ge(target)
    }
    
    /// 找到最近的大于 target 的键，返回索引
    ///
    pub fn find_index_gt(&self, target: K) -> set::FindResult<usize> {
        self.map.find_index_gt(target)
    }
    
    pub fn get_index(
        &self,
        target: K,
    ) -> set::GetIndexResult<usize>
    {
        self.map.get_index(target)
    }
    
    /// 计算指定值对应的块索引，但是带通用前置检查，但检查不通过时panic
    ///
    /// 获取值对应的索引。
    ///
    /// # Panics
    /// 详见[`GetIndexRawFieldSetError`]
    pub fn unchecked_get_index(
        &self,
        target: K,
    ) -> usize
    {
        self.map.unchecked_get_index(target)
    }
    
}