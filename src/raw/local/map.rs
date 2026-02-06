use thiserror::Error;
use std::hash::Hash;
use std::mem;
use ahash::AHashMap;
use std::ops::{Div, Mul, Sub};
use num_traits::real::Real;
use span_core::Span;
use super::set::*;

pub(crate) type FindResult<T> = Result<T, FindRawFieldMapError>;

#[derive(Error, Debug)]
pub enum FindRawFieldMapError {
    #[error("目标值超出了当前RawFieldMap的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}


pub(crate) type ReplaceIndexResult<T> = Result<T, ReplaceIndexRawFieldMapError<T>>;

#[derive(Error, Debug)]
pub enum ReplaceIndexRawFieldMapError<T> {
    #[error("指定的块为空块")]
    EmptyField(T),
}

impl<T> ReplaceIndexRawFieldMapError<T>{
    pub fn unwrap(self) -> T {
        match self {
            Self::EmptyField(v) => {v}
        }
    }
}

pub(crate) type TryInsertResult<T> = Result<(), TryInsertRawFieldMapError<T>>;
#[derive(Error, Debug)]
pub enum TryInsertRawFieldMapError<T> {
    /// Key超出span范围（携带失败的 T）
    #[error("Key超出了当前RawFieldMap的span范围，插入的值：{0:?}")]
    OutOfSpan(T),
    /// Key已存在（携带失败的 T）
    #[error("Key对应块已存在元素，插入的值：{0:?}")]
    AlreadyExists(T),
}

impl<T> TryInsertRawFieldMapError<T>{
    pub fn unwrap(self) -> T {
        match self {
            Self::OutOfSpan(v) => {v}
            Self::AlreadyExists(v) => {v}
        }
    }
    
    pub(crate) fn from_set(value:T, err: TryInsertRawFieldSetError) -> Self {
        match err {
            TryInsertRawFieldSetError::OutOfSpan => {Self::OutOfSpan(value)}
            TryInsertRawFieldSetError::AlreadyExists => {Self::AlreadyExists(value)}
        }
    }
}

pub(crate) type InsertResult<K,V> = Result<Option<(K,V)>, InsertRawFieldMapError<V>>;
#[derive(Error, Debug)]
pub enum InsertRawFieldMapError<T> {
    /// Key超出span范围（携带失败的 T）
    #[error("Key超出了当前RawFieldMap的span范围，插入的值：{0:?}")]
    OutOfSpan(T),
}

impl<T> InsertRawFieldMapError<T>{
    pub fn unwrap(self) -> T {
        match self {
            Self::OutOfSpan(v) => {v}
        }
    }
    
    pub(crate) fn from_set(value:T ,err: InsertRawFieldSetError) -> Self {
        match err {
            InsertRawFieldSetError::OutOfSpan => {Self::OutOfSpan(value)}
        }
    }
}


/// 可使用Key快速查找值的升序序列。<br>
/// 因本质是分块思想，每个单元为以块，故命名为 `RawFieldMap`
///
/// 仅单线程下使用
///
/// 将一块区域划分成小块，每一块只能存入**一个或零个**元素 <br>
/// 第n个（从0开始）块代表其中的元素处于`[n*unit,(n+1)unit) (n∈N)`区间 <br>
/// 若没有元素处于一个块，此块称为空块，这个块以某种方式存 前一个非空块 与 后一个非空块 数据的 引用 <br>
/// 如果每一个块都是空块，每个块不存储任何内容，此容器视为空，但内存占用不一定只有span和unit（见 [`Self::capacity`] ）<br>
///
/// 适合极度频繁查找+删增插较不频繁的场景<br>
/// 实现了O(1)的查找，同时删增插皆为O(m) <br>
/// 同时有O(m)空间复杂度 <br>
/// 其中`m ∈ Z, m <= Self::size()`（除以后向上取整），即m为当前块数量（见 [`Self::len`] ）
///
/// 最大块数量为`(span/unit).into() + 1`
///
#[derive(Default, Debug)]
pub struct RawFieldMap<K,V>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + Into<usize> + Sized + Real,
    K: Hash + Eq,
{
    pub(crate) keys: RawFieldSet<K>,
    pub(crate) values: AHashMap<K,V>
}

impl<K,V> RawFieldMap<K,V>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + Into<usize> + Sized + Real,
    K: Hash + Eq,
{
    /// 提供span与unit，构建一个RawFieldMap
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<K>, unit: K) -> Result<Self, (Span<K>, K)> {
        Ok(Self {
            keys: RawFieldSet::new(span, unit)?,
            values: AHashMap::new()
        })
    }
    
    /// 提供span与unit，构建一个RawFieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0、span为空、capacity大于最大块数量，通过返回Err返还提供的数据
    pub fn with_capacity(span: Span<K>, unit: K, capacity: usize) -> Result<Self, (Span<K>, K)> {
        Ok(Self {
            keys: RawFieldSet::with_capacity(span, unit, capacity)?,
            values: AHashMap::new()
        })
    }
    
    /// 通过索引得到块值引用
    ///
    /// 若块不为空，返回Some
    /// 
    /// # Panics
    /// 插入逻辑确保存在Key时就存在value。若存在对应Key而不存在对应Value，panic。
    pub fn get(&self, idx: usize) -> Option<&V> {
        Some(self.values.get(&self.keys.get(idx)?).unwrap())
    }
    
    /// 通过索引得到值引用
    ///
    /// 若块不为空，返回Some
    ///
    /// # Panics
    /// 越界访问时panic
    /// 
    /// 插入逻辑确保存在Key时就存在value。若存在对应Key而不存在对应Value，panic。
    pub fn unchecked_get(&self, idx: usize) -> Option<&V> {
        Some(self.values.get(&self.keys.get_in(idx)?).unwrap())
    }
    
    /// 尝试插入值
    ///
    /// 插入失败会返回 `TryInsertRawFieldSetError` ，使用 `unwrap` 方法得到传入值 `value`。
    pub fn try_insert(&mut self, key: K, value: V) -> TryInsertResult<V> {
        match self.keys.try_insert(key) {
            Ok(v) => { v }
            // 不能用map_err，因为需要拿到value的所有权。
            Err(err) => {
                return Err(TryInsertRawFieldMapError::from_set(value, err))
            }
        }
        self.values.insert(key, value);
        Ok(())
    }
    
    /// 插入或替换值
    ///
    /// 若对应块已有值，新值将替换原值，返回Ok(Some((K,V)))包裹原键值。<br>
    /// 若无值，插入新值返回 Ok(None)。
    ///
    pub fn insert(&mut self, key: K, value: V) -> InsertResult<K, V> {
        let result =
            match self.keys.insert(key) {
                Ok(v) => { v }
                // 不能用map_err，因为需要拿到value的所有权。
                Err(err) => {
                    return Err(InsertRawFieldMapError::from_set(value, err))
                }
            }
                // 上面确保Some时是已存在k，所以unwrap没问题。panic就是别的地方逻辑有问题！
                .map(|k| (k, self.values.remove(&k).unwrap()))
            ;
        self.values.insert(key, value);
        Ok(result)
    }
    
    
    /// 用索引指定替换块的键
    ///
    /// 成功则返回其原键
    pub fn replace_key_index(&mut self, idx: usize, key: K) -> super::set::ReplaceIndexResult<K>
    where
        K: Mul<usize, Output = K>,
    {
        self.keys.replace_index(idx,key)
    }
    
    /// 用索引指定替换块的键，但不进行索引检查
    ///
    /// 成功则返回其原键
    ///
    /// # Panics
    /// 索引越界时panic
    pub(crate) fn replace_key_index_in(&mut self, idx: usize, key: K) -> super::set::ReplaceIndexResult<K>
    where
        K: Mul<usize, Output = K>,
    {
        self.keys.replace_index_in(idx, key)
    }
    
    /// 用索引指定替换块的键，但无法替换时panic
    ///
    /// 返回其原键
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    pub(crate) fn unchecked_replace_key_index(&mut self, idx: usize, key: K) -> K
    where
        K: Mul<usize, Output = K> + std::fmt::Debug,
    {
        self.keys.unchecked_replace_index(idx, key)
    }
    
    
    /// 用索引指定替换块的值
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 插入逻辑确保存在Key时就存在value。若存在对应Key而不存在对应Value，panic。
    pub fn replace_index(&mut self, idx: usize, value: V) -> ReplaceIndexResult<V> {
        use ReplaceIndexRawFieldMapError::*;
        
        let items = &mut self.keys.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField(value)) }
        
        self.replace_index_in(idx, value)
    }
    
    /// 用索引指定替换块的值，但不进行索引检查
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 插入逻辑确保存在Key时就存在value。若存在对应Key而不存在对应Value，panic。
    pub(crate) fn replace_index_in(&mut self, idx: usize, value: V) -> ReplaceIndexResult<V> {
        use ReplaceIndexRawFieldMapError::*;
        
        if let RawField::Thing(thing) = self.keys.items[idx] {
            Ok(mem::replace(&mut self.values.get_mut(&thing.1).unwrap(),value))
        } else {
            Err(EmptyField(value))
        }
    }
    
    /// 用索引指定替换块的值，但无法替换时panic
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    ///
    /// 插入逻辑确保存在Key时就存在value。若存在对应Key而不存在对应Value，panic。
    pub fn unchecked_replace_index(&mut self, idx: usize, value: V) -> V {
        if let RawField::Thing(thing) = self.keys.items[idx] {
            mem::replace(&mut self.values.get_mut(&thing.1).unwrap(),value)
        } else {
            panic!("Called `RawFieldMap::unchecked_replace_index()` on a empty field")
        }
    }
    
    
}

impl<K,V> Deref for RawFieldMap<K,V>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real ,
    K: Hash + Eq,
{
    type Target = RawFieldSet<K>;
    
    /// 只有一些方法并需要进行上层包装。比如插入相关。
    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}

impl<K,V> DerefMut for RawFieldMap<K,V>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real ,
    K: Hash + Eq,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.keys
    }
}
