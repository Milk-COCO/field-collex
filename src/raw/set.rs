use span_core::Span;
use num_traits::real::Real;
use std::mem;
use std::ops::Mul;
use std::vec::Vec;
use thiserror::Error;

/// 一个块。详见 具体容器类型 。
///
/// Thing：本块有元素，存本块索引+值 <br>
/// Prev ：本块无元素，有前一个非空块，存其索引 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者索引 <br>
/// Next ：本块无元素，有后一个非空块，存其索引 <br>
/// Void ：容器完全无任何元素 <br>
///
#[derive(Debug, Clone)]
pub enum RawField<V> {
    Thing((usize, V)),
    Prev (usize),
    Among(usize, usize),
    Next (usize),
    Void,
}

impl<V> RawField<V> {
    pub fn as_thing(&self) -> &(usize, V) {
        match self {
            Self::Thing(t) => t,
            _ => panic!("Called `RawField::as_thing()` on a not `Thing` value`"),
        }
    }
    
    pub fn void() -> Self {
        Self::Void
    }
    
    /// 从Thing制造Prev
    ///
    /// # Panics
    /// 非Thing时Panic
    pub fn make_prev(&self) -> Self {
        match self {
            Self::Thing(t) => Self::Prev(t.0),
            _ => panic!("Called `RawField::make_prev()` on a not `Thing` value`"),
        }
    }
    
    /// 从Thing制造Next
    ///
    /// # Panics
    /// 非Thing时Panic
    pub fn make_next(&self) -> Self {
        match self {
            Self::Thing(t) => Self::Next(t.0),
            _ => panic!("Called `RawField::make_next()` on a not `Thing` value`"),
        }
    }
    
    pub fn prev_from(tuple: &(usize, V)) -> Self {
        Self::Prev(tuple.0)
    }
    
    pub fn next_from(tuple: &(usize, V)) -> Self {
        Self::Next(tuple.0)
    }
    
    
    /// 得到当前或上一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    pub fn thing_prev(&self) -> Option<usize> {
        match self {
            RawField::Thing(v) => Some(v.0),
            RawField::Prev(prev)
            | RawField::Among(prev,..)
            => Some(*prev),
            _ => None,
        }
    }
    
    
    /// 得到当前或下一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    pub fn thing_next(&self) -> Option<usize> {
        match self {
            RawField::Thing(v) => Some(v.0),
            RawField::Next(next)
            | RawField::Among(_, next)
            => Some(*next),
            _ => None,
        }
    }
    
}

pub(crate) type NewResult<T,V> = Result<T, NewRawFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum NewRawFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    ZeroUnit(Span<V>, V),
}

impl<V> NewRawFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::ZeroUnit(span, unit)
            | Self::EmptySpan(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type WithCapacityResult<T,V> = Result<T, WithCapacityRawFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum WithCapacityRawFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    ZeroUnit(Span<V>, V),
    #[error("提供的 capacity 超过最大块数量")]
    OutOfSize(Span<V>, V),
}

impl<V> WithCapacityRawFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::ZeroUnit(span, unit)
            | Self::EmptySpan(span, unit)
            | Self::OutOfSize(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type FindMatcherResult<T> = Result<T, FindMatcherRawFieldSetError>;

#[derive(Error, Debug)]
pub(crate) enum FindMatcherRawFieldSetError {
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}

impl From<FindMatcherRawFieldSetError> for FindRawFieldSetError{
    fn from(value: FindMatcherRawFieldSetError) -> Self {
        match value { 
            FindMatcherRawFieldSetError::CannotFind => {Self::CannotFind}
            FindMatcherRawFieldSetError::Empty => {Self::Empty}
        }
    }
}

pub(crate) type GetIndexResult<T> = Result<T, GetIndexRawFieldSetError>;

#[derive(Error, Debug)]
pub enum GetIndexRawFieldSetError {
    #[error("目标值超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("当前无数据可查询")]
    Empty,
}

macro_rules! impl_from_get_index_err {
    ($err: ident) => {
        impl From<GetIndexRawFieldSetError> for $err{
            fn from(value: GetIndexRawFieldSetError) -> Self {
                match value {
                    GetIndexRawFieldSetError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexRawFieldSetError::Empty => {Self::Empty}
                }
            }
        }
    };
    ($err: ident, $empty: ident) => {
        impl From<GetIndexRawFieldSetError> for $err{
            fn from(value: GetIndexRawFieldSetError) -> Self {
                match value {
                    GetIndexRawFieldSetError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexRawFieldSetError::Empty => {Self::$empty}
                }
            }
        }
    };
}

impl_from_get_index_err!(FindRawFieldSetError);
impl_from_get_index_err!(ReplaceRawFieldSetError, EmptyField);
impl_from_get_index_err!(RemoveRawFieldSetError, EmptyField);


pub(crate) type FindResult<T> = Result<T, FindRawFieldSetError>;

#[derive(Error, Debug)]
pub enum FindRawFieldSetError {
    #[error("目标值超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}

pub(crate) type ReplaceIndexResult<T> = Result<T, ReplaceIndexRawFieldSetError>;

#[derive(Error, Debug)]
pub enum ReplaceIndexRawFieldSetError {
    #[error("指定的块为空块")]
    EmptyField,
    #[error("提供的值不属于此区间")]
    OutOfField,
}


pub(crate) type RemoveIndexResult<T> = Result<T, RemoveIndexRawFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveIndexRawFieldSetError {
    #[error("指定的块已为空块")]
    EmptyField,
}


pub(crate) type ReplaceResult<T> = Result<T, ReplaceRawFieldSetError>;

#[derive(Error, Debug)]
pub enum ReplaceRawFieldSetError {
    #[error("提供的值超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("指定的块为空块")]
    EmptyField,
    #[error("提供的值不属于此区间")]
    OutOfField,
}

impl From<ReplaceIndexRawFieldSetError> for ReplaceRawFieldSetError {
    fn from(value: ReplaceIndexRawFieldSetError) -> Self {
        match value {
            ReplaceIndexRawFieldSetError::EmptyField => {Self::EmptyField}
            ReplaceIndexRawFieldSetError::OutOfField => {Self::OutOfField}
        }
    }
}

pub(crate) type RemoveResult<T> = Result<T, RemoveRawFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveRawFieldSetError {
    #[error("目标值超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("指定的块已为空块")]
    EmptyField,
}

impl From<RemoveIndexRawFieldSetError> for RemoveRawFieldSetError {
    fn from(value: RemoveIndexRawFieldSetError) -> Self {
        match value {
            RemoveIndexRawFieldSetError::EmptyField => {Self::EmptyField}
        }
    }
}


pub(crate) type TryInsertResult = Result<(), TryInsertRawFieldSetError>;
#[derive(Error, Debug)]
pub enum TryInsertRawFieldSetError {
    #[error("Key超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("Key对应块已存在元素")]
    AlreadyExists,
}

pub(crate) type InsertResult<V> = Result<Option<V>, InsertRawFieldSetError>;
#[derive(Error, Debug)]
pub enum InsertRawFieldSetError {
    #[error("Key超出了当前RawFieldSet的span范围")]
    OutOfSpan,
}

/// O(1)根据upper_bound或lower_bound查找值
/// 
#[derive(Default, Debug)]
pub struct RawFieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    pub(crate) span: Span<V>,
    pub(crate) unit: V,
    pub(crate) items: Vec<RawField<V>>,
    
}

impl<V> RawFieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    /// 提供span与unit，构建一个RawFieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<V>, unit: V) -> NewResult<Self,V> {
        use NewRawFieldSetError::*;
        
        if unit.is_zero() {
            Err(ZeroUnit(span, unit))
        } else if span.is_empty() {
            Err(EmptySpan(span, unit))
        } else {
            Ok(Self {
                span,
                unit,
                items: Vec::new()
            })
        }
    }
    
    /// 提供span与unit，构建一个RawFieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0、span为空、capacity大于最大块数量，通过返回Err返还提供的数据
    pub fn with_capacity(span: Span<V>, unit: V, capacity: usize) -> WithCapacityResult<Self,V> {
        use WithCapacityRawFieldSetError::*;
        if unit.is_zero() {
            Err(ZeroUnit(span, unit))
        } else if span.is_empty() {
            Err(EmptySpan(span, unit))
        } else if match span.size(){
            Ok(Some(size)) => {
                capacity > (size / unit).ceil().into()
            },
            Ok(None) => {false}
            // is_empty为真时，永远不可能出现Err，因为它绝对有长度
            _ => unreachable!()
        } {
            Err(OutOfSize(span, unit))
        } else {
            Ok(Self {
                span,
                unit,
                items: Vec::with_capacity(capacity),
            })
        }
    }
    
    // /// 尝试从内部数据构建一个 `RawFieldSet`
    // ///
    // /// 若构建失败，返回传入值（`Err(Span<V>, V, Vec<V>`）。
    // pub fn try_from_inner(span: Span<V>, unit: V, items: Vec<V>) -> Result<Self,(Span<V>, V, Vec<V>)> {
    //     if span.size() < unit {
    //         Err((span, unit, items))
    //     } else {
    //         Ok(Self { span, unit, items })
    //     }
    // }
    
    pub fn span(&self) -> &Span<V> {
        &self.span
    }
    
    pub fn unit(&self) -> &V {
        &self.unit
    }
    
    /// 返回最大块数量
    ///
    /// 若Span是无限区间，返回None
    pub fn size(&self) -> Option<usize> {
        // 确保在创建时就不可能为空区间。详见那些构造函数
        match self.span.size(){
            Ok(Some(size)) => Some((size / self.unit).ceil().into()),
            _ => None
        }
    }
    
    /// 返回已存在的块的数量
    ///
    /// 此值应小于等于最大块数量
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    /// Returns the total number of elements the inner vector can hold without reallocating.
    ///
    /// 此值应小于等于最大块数量
    pub fn capacity(&self) -> usize {
        self.items.capacity()
    }
    
    /// 判断对应块是非空
    pub fn is_thing(&self, idx: usize) -> bool {
        if idx < self.items.len() {
            matches!(self.items[idx], RawField::Thing(_))
        } else { false }
    }
    
    /// 通过索引返回块内数据
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub fn thing(&self, idx: usize) -> Option<V> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref v) => Some(v.1),
                _ => None
            }
        } else { None }
    }
    
    /// 通过索引返回块内数据引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub fn thing_ref(&self, idx: usize) -> Option<&V> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref v) => Some(&v.1),
                _ => None
            }
        } else { None }
    }
    
    /// 通过索引返回块内数据可变引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub fn thing_mut(&mut self, idx: usize) -> Option<&mut V> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref mut v) => Some(&mut v.1),
                _ => None
            }
        } else { None }
    }
    
    /// 计算指定值对应的块索引
    ///
    /// 此无任何前置检查，只会机械地返回目标相对于初始位置（区间的左端点）可能处于第几个块，但不确保这个块是否合法。<br>
    /// 包含前置检查的版本是[`get_index`]
    #[inline(always)]
    pub fn idx_of(&self, target: V) -> usize {
        ((target - *self.span.start()) / self.unit).into()
    }
    
    pub(crate) fn resize_to_idx(&mut self, idx: usize) {
        if self.items.len() <= idx {
            let filler = match self.items.last() {
                None =>
                    RawField::Void,
                Some(last) => {
                    if let RawField::Thing(t) = last {
                        RawField::prev_from(t)
                    } else {
                        last.clone()
                    }
                }
            };
            self.items.resize(idx+1,filler);
        }
    }
    
    /// 查找对应值是否存在
    ///
    pub fn contains(&self, value: V) -> bool {
        matches!(self.items[self.idx_of(value)], RawField::Thing((_,k)) if k == value)
    }
    
    pub fn get_field(&self, idx: usize) -> Option<&RawField<V>> {
        if idx < self.items.len() {
            Some(&self.items[idx])
        } else {None}
    }
    
    /// 通过索引得到当前或上一个非空块的(索引,值)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，或索引越界，返回None <br>
    pub fn get_prev(&self, idx: usize) -> Option<(usize,V)> {
        Some(*self.items[self.get_prev_index(idx)?].as_thing())
    }
    
    /// 通过索引得到当前或下一个非空块的(索引,值)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，或索引越界，返回None <br>
    pub fn get_next(&self,idx: usize) -> Option<(usize,V)> {
        Some(*self.items[self.get_next_index(idx)?].as_thing())
    }
    
    
    /// 通过索引得到当前或上一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，或索引越界，返回None <br>
    pub fn get_prev_index(&self, idx: usize) -> Option<usize> {
        if idx < self.items.len() {
            self.items[idx].thing_prev()
        } else { None }
    }
    
    /// 通过索引得到当前或下一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，或索引越界，返回None <br>
    pub fn get_next_index(&self,idx: usize) -> Option<usize> {
        if idx < self.items.len() {
            self.items[idx].thing_next()
        } else { None }
    }
    
    
    /// 辅助函数：执行插入/替换后的前后更新逻辑
    pub(crate) fn try_insert_in(
        &mut self,
        idx: usize,
        target: V,
    ) {
        // 扩容到目标索引
        self.resize_to_idx(idx);
        
        let items = &mut self.items;
        
        let len = items.len();
        
        // 向后更新
        items[idx+1..len].iter_mut()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                *v = match *v {
                    RawField::Prev(next) | RawField::Among(_, next) => RawField::Among(idx, next),
                    RawField::Next(_) | RawField::Void => RawField::Next(idx),
                    _ => unreachable!()
                };
            });
        
        // 向前更新
        items[0..idx].iter_mut().rev()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                *v = match *v {
                    RawField::Next(prev) | RawField::Among(prev, _) => RawField::Among(prev, idx),
                    RawField::Prev(_) | RawField::Void => RawField::Prev(idx),
                    _ => unreachable!()
                };
            });
        
        items[idx] = RawField::Thing((idx,target));
    }
    
    /// 尝试插入值
    ///
    /// 插入失败会返回 [`TryInsertRawFieldSetError`]
    pub fn try_insert(&mut self, target: V) -> TryInsertResult {
        use TryInsertRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan) }
        
        let idx = self.idx_of(target);
        
        if self.is_thing(idx) {return Err(AlreadyExists)};
        
        self.try_insert_in(
            idx,
            target,
        );
        Ok(())
    }
    
    /// 插入或替换值
    ///
    /// 若对应块已有值，新值将替换原值，返回Ok(Some(V))包裹原值。<br>
    /// 若无值，插入新值返回None。
    ///
    pub fn insert(&mut self, target: V) -> InsertResult<V>  {
        use InsertRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan) }
        
        let idx = self.idx_of(target);
        
        if let Some(thing) = self.thing_mut(idx){
            // 已存在，则替换并返回其原值
            let old = *thing;
            *thing = target;
            Ok(Some(old))
        } else {
            // 同 try_insert
            self.try_insert_in(
                idx,
                target,
            );
            
            Ok(None)
        }
    }
    
    
    /// 用索引指定替换块
    ///
    /// 成功则返回其原值
    pub fn replace_index(&mut self, idx: usize, target: V) -> ReplaceIndexResult<V>
    where
        V: Mul<usize, Output = V>,
    {
        use ReplaceIndexRawFieldSetError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField) }
        
        self.replace_index_in(idx, target)
    }
    
    /// 用索引指定替换块，但不进行索引检查
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    pub(crate) fn replace_index_in(&mut self, idx: usize, target: V) -> ReplaceIndexResult<V>
    where
        V: Mul<usize, Output = V>,
    {
        use ReplaceIndexRawFieldSetError::*;
        
        if !(self.unit*idx..self.unit*(idx+1)).contains(&target) {
            return Err(OutOfField);
        }
        
        if let RawField::Thing(ref mut thing) = self.items[idx] {
            Ok(mem::replace(&mut (thing.1),target ))
        } else {
            Err(EmptyField)
        }
    }
    
    /// 用索引指定替换块，但无法替换时panic
    ///
    /// 返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    ///
    /// 指定块为空时panic
    pub fn unchecked_replace_index(&mut self, idx: usize, target: V) -> V
    where
        V: Mul<usize, Output = V> + std::fmt::Debug,
    {
        let range = self.unit*idx..self.unit*(idx+1);
        if !range.contains(&target) {
            panic!("Called `RawField::unchecked_replace_index()` with target '{target:?}' cannot be in field {idx} (out of {range:?})")
        }
        // 提示编译器这个range后面不再用到
        drop(range);
        
        if let RawField::Thing(ref mut thing) = self.items[idx] {
            mem::replace(&mut (thing.1),target)
        } else {
            panic!("Called `RawField::unchecked_replace_index()` on a empty field")
        }
    }
    
    
    /// 用索引指定清空块。
    ///
    /// 若指定块非空，返回内部值。
    pub fn remove_index(&mut self, idx: usize) -> RemoveIndexResult<V>
    {
        use RemoveIndexRawFieldSetError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField) }
        
        self.remove_index_in(idx)
    }
    
    /// 用索引指定清空块，但不进行索引检查
    ///
    /// 若指定块非空，返回内部值。
    ///
    /// # Panics
    /// 索引越界时panic
    pub(crate) fn remove_index_in(&mut self, idx: usize) -> RemoveIndexResult<V> {
        use RemoveIndexRawFieldSetError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if let RawField::Thing(_) = items[idx] {
            // 根据上一个元素与下一个元素，生成填充元素
            let next =
                if idx == len-1 {
                    None
                } else {
                    match &items[idx + 1] {
                        RawField::Thing(thing) => Some(thing.0),
                        RawField::Prev(_)
                        | RawField::Void => None,
                        RawField::Among(_, next)
                        | RawField::Next(next) => Some(*next),
                    }
                };
            
            let prev =
                if idx == 0 {
                    None
                } else {
                    match &items[idx - 1] {
                        RawField::Thing(thing) => Some(thing.0),
                        RawField::Next(_)
                        | RawField::Void => None,
                        RawField::Among(prev, _)
                        | RawField::Prev(prev) => Some(*prev),
                    }
                };
            
            let maker = ||
                match next {
                    None =>
                        match prev {
                            None => RawField::Void,
                            Some(prev) => RawField::Prev(prev),
                        },
                    Some(next) =>
                        match prev {
                            None => RawField::Next(next),
                            Some(prev) => RawField::Among(prev, next),
                        },
                };
            
            // 更新自己
            let old = mem::replace(&mut items[idx], maker());
            
            // 向前更新
            items[0..idx].iter_mut().rev()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    *v = maker();
                });
            
            // 向后更新
            items[idx+1..len].iter_mut()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    *v = maker();
                });
            
            // 刚刚更新的过程中已确保不再存在任何自己的借用，直接unwrap！
            Ok(old.as_thing().1)
        } else {
            Err(EmptyField)
        }
    }
    
    /// 用索引指定清空块，但无法清空时panic
    ///
    /// 返回原值。
    ///
    /// # Panics
    /// 索引越界时panic
    /// 
    /// 指定块为空时panic
    pub fn unchecked_remove_index(&mut self, idx: usize) -> V {
        if let Ok(v) = self.remove_index_in(idx) {
            v
        } else {
            panic!("Called `RawField::unchecked_remove_index()` on a empty field")
        }
    }
    
    
    /// 替换指定值对应的指定块
    ///
    /// 成功则返回其原值
    pub fn replace(&mut self, target: V) -> ReplaceResult<V>
    where
        V: Mul<usize, Output = V>,
    {
        let idx = self.get_index(target)
            .map_err(Into::<ReplaceRawFieldSetError>::into)?;
        
        self.replace_index_in(idx, target)
            .map_err(Into::<ReplaceRawFieldSetError>::into)
    }
    
    /// 替换指定值对应的指定块，但无法替换时panic
    ///
    /// 返回其原值
    ///
    /// # Panics
    /// 见[`unchecked_get_index`]和[`unchecked_replace_index`]
    pub fn unchecked_replace(&mut self, target: V) -> V
    where
        V: Mul<usize, Output = V> + std::fmt::Debug,
    {
        let idx = self.unchecked_get_index(target);
        
        self.unchecked_replace_index(idx,target)
    }
    
    
    /// 用值清空对应块。
    ///
    /// 若指定块非空，返回内部值。
    pub fn remove(&mut self, target: V) -> RemoveResult<V>
    {
        let idx = self.get_index(target)
            .map_err(Into::<RemoveRawFieldSetError>::into)?;
        
        self.remove_index_in(idx)
            .map_err(Into::<RemoveRawFieldSetError>::into)
    }
    
    /// 用值指定清空对应块，但无法清空时panic
    ///
    /// 返回原值。
    ///
    /// # Panics
    /// 同[`unchecked_get_index`] + [`unchecked_remove_index`]
    pub fn unchecked_remove(&mut self, target: V) -> V
    {
        let idx = self.unchecked_get_index(target);
        
        self.unchecked_remove_index(idx)
    }
    
    
    /// 找到最近的小于等于 target 的值
    ///
    pub fn find_le(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            Self::matcher_l,
            |the, tgt| *the <= *tgt,
            |idx,_| idx == 0,
            |idx| idx-1
        )
    }
    
    /// 找到最近的小于 target 的值
    ///
    pub fn find_lt(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            Self::matcher_l,
            |the, tgt| *the < *tgt,
            |idx,_| idx == 0,
            |idx| idx-1
        )
    }
    
    /// 找到最近的大于等于 target 的值
    ///
    pub fn find_ge(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            Self::matcher_r,
            |the,tgt| *the >= *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
    
    /// 找到最近的大于 target 的值
    ///
    pub fn find_gt(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            Self::matcher_r,
            |the,tgt| *the > *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
    
    /// 找到最近的小于等于 target 的值的索引
    ///
    pub fn find_index_le(&self, target: V) -> FindResult<usize> {
        self.find_index_in(
            target,
            Self::matcher_l,
            |the, tgt| *the <= *tgt,
            |idx,_| idx == 0,
            |idx| idx-1
        )
    }
    
    /// 找到最近的小于 target 的值的索引
    ///
    pub fn find_index_lt(&self, target: V) -> FindResult<usize> {
        self.find_index_in(
            target,
            Self::matcher_l,
            |the, tgt| *the < *tgt,
            |idx,_| idx == 0,
            |idx| idx-1
        )
    }
    
    /// 找到最近的大于等于 target 的值的索引
    ///
    pub fn find_index_ge(&self, target: V) -> FindResult<usize> {
        self.find_index_in(
            target,
            Self::matcher_r,
            |the,tgt| *the >= *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
    
    /// 找到最近的大于 target 的值的索引
    ///
    pub fn find_index_gt(&self, target: V) -> FindResult<usize> {
        self.find_index_in(
            target,
            Self::matcher_r,
            |the,tgt| *the > *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
    
    /// 计算指定值对应的块索引，但是带通用前置检查
    ///
    /// 获取值对应的索引。
    pub fn get_index(
        &self,
        target: V,
    ) -> GetIndexResult<usize> {
        use GetIndexRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan); }
        let items = &self.items;
        let len = items.len();
        if len == 0 { return Err(Empty); }
        
        Ok(self.idx_of(target).min(len - 1))
    }
    
    /// 计算指定值对应的块索引，但是带通用前置检查，但检查不通过时panic
    ///
    /// 获取值对应的索引。
    ///
    /// # Panics
    /// 详见[`GetIndexRawFieldSetError`]
    pub fn unchecked_get_index(
        &self,
        target: V,
    ) -> usize
    {
        let span = &self.span;
        if !span.contains(&target) { panic!("Called `RawFieldSet::unchecked_get_index()` on a target is not contained in the span"); }
        let items = &self.items;
        let len = items.len();
        if len == 0 { panic!("Called `RawFieldSet::unchecked_get_index()` on a empty set"); }
        
        self.idx_of(target).min(len - 1)
    }
    
    /// 通用底层查找核心
    ///
    /// # 参数
    /// - target: 查找目标值
    /// - matcher: 字段匹配器，解耦左右查找的字段匹配逻辑，入参为数据数组+索引，返回匹配的(idx, V)
    /// - cmp: 匹配判定规则  | (当前K, 目标K) -> bool | true = 命中当前项，直接返回V
    /// - lmt: 边界兜底规则  | (当前索引, 数组长度) -> bool | true = 触达边界，返回None
    /// - next: 索引跳转规则 | (当前索引) -> usize | 返回查找目标索引
    ///
    /// 因为查找是O(1)所以暂不使用迭代器
    pub(crate) fn find_in(
        &self,
        target: V,
        matcher: impl Fn(&Self, &RawField<V>) -> FindMatcherResult<(usize, V)>,
        cmp: impl FnOnce(&V,&V) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<V>
    {
        use FindRawFieldSetError::*;
        
        let idx = self.get_index(target)
            .map_err(Into::<FindRawFieldSetError>::into)?;
        let items = &self.items;
        let len = items.len();
        let current = matcher(self,&items[idx])
            .map_err(Into::<FindRawFieldSetError>::into)?;
        
        Ok(if cmp(&current.1, &target) {
            current.1
        } else {
            if lmt(idx, len) { return Err(CannotFind); }
            let next = matcher(self,&items[next(idx)])?;
            next.1
        })
    }
    
    pub(crate) fn find_index_in(
        &self,
        target: V,
        matcher: impl Fn(&Self, &RawField<V>) -> FindMatcherResult<(usize, V)>,
        cmp: impl FnOnce(&V,&V) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<usize>
    {
        use FindRawFieldSetError::*;
        let idx = self.get_index(target)
            .map_err(Into::<FindRawFieldSetError>::into)?;
        let items = &self.items;
        let len = items.len();
        let current = matcher(self,&items[idx])
            .map_err(Into::<FindRawFieldSetError>::into)?;
        
        Ok(if cmp(&current.1, &target) {
            idx
        } else {
            if lmt(idx, len) { return Err(CannotFind); }
            next(idx)
        })
    }
    
    pub(crate) fn matcher_l(this: &Self, field: &RawField<V>) -> FindMatcherResult<(usize, V)> {
        use FindMatcherRawFieldSetError::*;
        Ok(match field {
            RawField::Thing(thing)
            =>  (thing.0,thing.1),
            RawField::Prev(fount)
            | RawField::Among(fount, _)
            => {
                // fount必然是Thing。因为本来就是存的Thing啊。
                let thing = this.items[*fount].as_thing();
                (thing.0,thing.1)
            }
            RawField::Next(_)
            => return Err(CannotFind),
            RawField::Void
            => return Err(Empty),
        })
    }
    
    pub(crate) fn matcher_r(this: &Self, field: &RawField<V>) -> FindMatcherResult<(usize, V)> {
        use FindMatcherRawFieldSetError::*;
        Ok(match field {
            RawField::Thing(thing)
            =>  (thing.0,thing.1),
            RawField::Prev(_)
            => return Err(CannotFind),
            RawField::Next(next)
            | RawField::Among(_, next)
            => {
                // next必然是Thing。因为本来就是存的Thing啊。
                let thing = this.items[*next].as_thing();
                (thing.0,thing.1)
            }
            RawField::Void
            => return Err(Empty),
        })
    }
}
