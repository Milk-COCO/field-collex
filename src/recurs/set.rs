use span_core::Span;
use num_traits::real::Real;
use std::mem;
use std::ops::Mul;
use std::vec::Vec;
use thiserror::Error;
use crate::FieldItem;

/// 一个块。详见 具体容器类型 。
///
/// Thing：本块有元素，存本块索引+值 <br>
/// Prev ：本块无元素，有前一个非空块，存其索引 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者索引 <br>
/// Next ：本块无元素，有后一个非空块，存其索引 <br>
/// Void ：容器完全无任何元素 <br>
///
#[derive(Debug)]
pub enum RawField<V> {
    Thing((usize, V)),
    Prev (usize),
    Among(usize, usize),
    Next (usize),
    Void,
}

impl<V> RawField<V> {
    pub fn as_thing(&self) -> (usize, &V) {
        match self {
            Self::Thing(t) => (t.0,&t.1),
            _ => panic!("Called `RawField::as_thing()` on a not `Thing` value`"),
        }
    }
    
    /// 得到Thing内部值
    /// 
    /// # Panics
    /// 非Thing时panic
    pub fn unwrap(self) -> V {
        match self {
            Self::Thing(t) => t.1,
            _ => panic!("Called `RawField::unwrap()` on a not `Thing` value`"),
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
    
    pub fn partial_clone(&self) -> Option<Self> {
        match *self {
            RawField::Thing(_) => None,
            _ => Some(match *self {
                RawField::Prev(p) => RawField::Prev(p),
                RawField::Among(p, n) => RawField::Among(p, n),
                RawField::Next(n) => RawField::Next(n),
                RawField::Void => RawField::Void,
                _ => unreachable!(),
            }),
        }
    }
}

impl<V> Clone for RawField<V> {
    /// # Panics
    /// 为Thing时panic
    fn clone(&self) -> Self {
        self.partial_clone().expect("Called `RawField::clone` on a `Thing` value")
    }
}

type FieldIn<V> = Field<V,FieldSet<V>>;
type SetField<V> = RawField<Field<V,FieldSet<V>>>;

#[derive(Debug)]
pub enum Field<V,C>{
    Elem(V),
    Collex(C)
}

impl<V> FieldItem<V> for Field<V,FieldSet<V>>
where
    V: Ord + Real + Into<usize>,
{
    fn first(&self) -> V {
        match self{
            Field::Elem(e) => {*e}
            Field::Collex(set) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                set.first().unwrap()
            }
        }
    }
    
    fn last(&self) -> V {
        match self{
            Field::Elem(e) => {*e},
            Field::Collex(set) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                set.last().unwrap()
            }
        }
    }
    
}

pub(crate) type NewResult<T,V> = Result<T, NewFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum NewFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    ZeroUnit(Span<V>, V),
}

impl<V> NewFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::ZeroUnit(span, unit)
            | Self::EmptySpan(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type WithCapacityResult<T,V> = Result<T, WithCapacityFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum WithCapacityFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    ZeroUnit(Span<V>, V),
    #[error("提供的 capacity 超过最大块数量")]
    OutOfSize(Span<V>, V),
}

impl<V> WithCapacityFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::ZeroUnit(span, unit)
            | Self::EmptySpan(span, unit)
            | Self::OutOfSize(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type FindMatcherResult<T> = Result<T, FindMatcherFieldSetError>;

#[derive(Error, Debug)]
pub(crate) enum FindMatcherFieldSetError {
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}

impl From<FindMatcherFieldSetError> for FindFieldSetError{
    fn from(value: FindMatcherFieldSetError) -> Self {
        match value {
            FindMatcherFieldSetError::CannotFind => {Self::CannotFind}
            FindMatcherFieldSetError::Empty => {Self::Empty}
        }
    }
}

pub(crate) type GetIndexResult<T> = Result<T, GetIndexFieldSetError>;

#[derive(Error, Debug)]
pub enum GetIndexFieldSetError {
    #[error("目标值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("当前无数据可查询")]
    Empty,
}

macro_rules! impl_from_get_index_err {
    ($err: ident) => {
        impl From<GetIndexFieldSetError> for $err{
            fn from(value: GetIndexFieldSetError) -> Self {
                match value {
                    GetIndexFieldSetError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexFieldSetError::Empty => {Self::Empty}
                }
            }
        }
    };
    ($err: ident, $empty: ident) => {
        impl From<GetIndexFieldSetError> for $err{
            fn from(value: GetIndexFieldSetError) -> Self {
                match value {
                    GetIndexFieldSetError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexFieldSetError::Empty => {Self::$empty}
                }
            }
        }
    };
}

impl_from_get_index_err!(FindFieldSetError);
impl_from_get_index_err!(ReplaceFieldSetError, EmptyField);
impl_from_get_index_err!(RemoveFieldSetError, EmptyField);


pub(crate) type FindResult<T> = Result<T, FindFieldSetError>;

#[derive(Error, Debug)]
pub enum FindFieldSetError {
    #[error("目标值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}

pub(crate) type ReplaceIndexResult<T> = Result<T, ReplaceIndexFieldSetError>;

#[derive(Error, Debug)]
pub enum ReplaceIndexFieldSetError {
    #[error("指定的块为空块")]
    EmptyField,
    #[error("提供的值不属于此区间")]
    OutOfField,
}


pub(crate) type RemoveIndexResult<T> = Result<T, RemoveIndexFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveIndexFieldSetError {
    #[error("指定的块已为空块")]
    EmptyField,
}


pub(crate) type ReplaceResult<T> = Result<T, ReplaceFieldSetError>;

#[derive(Error, Debug)]
pub enum ReplaceFieldSetError {
    #[error("提供的值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("指定的块为空块")]
    EmptyField,
    #[error("提供的值不属于此区间")]
    OutOfField,
}

impl From<ReplaceIndexFieldSetError> for ReplaceFieldSetError {
    fn from(value: ReplaceIndexFieldSetError) -> Self {
        match value {
            ReplaceIndexFieldSetError::EmptyField => {Self::EmptyField}
            ReplaceIndexFieldSetError::OutOfField => {Self::OutOfField}
        }
    }
}

pub(crate) type RemoveResult<T> = Result<T, RemoveFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveFieldSetError {
    #[error("目标值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("指定的块已为空块")]
    EmptyField,
}

impl From<RemoveIndexFieldSetError> for RemoveFieldSetError {
    fn from(value: RemoveIndexFieldSetError) -> Self {
        match value {
            RemoveIndexFieldSetError::EmptyField => {Self::EmptyField}
        }
    }
}


pub(crate) type TryInsertResult = Result<(), TryInsertFieldSetError>;
#[derive(Error, Debug)]
pub enum TryInsertFieldSetError {
    #[error("Key超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("Key对应块已存在元素")]
    AlreadyExists,
}

pub(crate) type InsertResult<V> = Result<Option<V>, InsertFieldSetError>;
#[derive(Error, Debug)]
pub enum InsertFieldSetError {
    #[error("Key超出了当前FieldSet的span范围")]
    OutOfSpan,
}


/// 上层包装。每个块可以存多个内容（通过递归结构实现）
/// 非空块可为单个元素或一个FieldSet，以[`Field`]类型存储。
#[derive(Default, Debug)]
pub struct FieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    pub(crate) span: Span<V>,
    pub(crate) unit: V,
    pub(crate) items: Vec<SetField<V>>,
    
}

impl<V> FieldSet<V>
where
    V: Ord + Real + Into<usize>,
{
    /// 提供span与unit，构建一个FieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<V>, unit: V) -> NewResult<Self,V> {
        use NewFieldSetError::*;
        
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
    
    /// 提供span与unit，构建一个FieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0、span为空、capacity大于最大块数量，通过返回Err返还提供的数据
    pub fn with_capacity(span: Span<V>, unit: V, capacity: usize) -> WithCapacityResult<Self,V> {
        use WithCapacityFieldSetError::*;
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
    pub(crate) fn is_thing(&self, idx: usize) -> bool {
        if idx < self.items.len() {
            matches!(self.items[idx], RawField::Thing(_))
        } else { false }
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
                        // 上面确保不是thing
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
        match &self.items[self.idx_of(value)]
        {
            RawField::Thing((_,k)) =>
                match k {
                    Field::Elem(e) => { value == *e }
                    Field::Collex(set) => { set.contains(value) }
                }
            _ => false
        }
    }
    
    /// 通过索引返回块引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub(crate) fn get_field(&self, idx: usize) -> Option<&FieldIn<V>> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref v) => Some(&v.1),
                _ => None
            }
        } else { None }
    }
    
    
    /// 通过索引得到当前或上一个非空块的(索引,块引用)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    /// 提供的索引大于最后一个块，相当于最后一个块 <br>
    pub(crate) fn get_prev_field(&self, idx: usize) -> Option<(usize,&FieldIn<V>)> {
        Some(self.items[self.get_prev_index(idx)?].as_thing())
    }
    
    /// 通过索引得到当前或下一个非空块的(索引,块引用)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    /// 提供的索引大于最后一个块，返回None <br>
    pub(crate) fn get_next_field(&self,idx: usize) -> Option<(usize,&FieldIn<V>)> {
        Some(self.items[self.get_next_index(idx)?].as_thing())
    }
    
    
    /// 通过索引得到当前或上一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    /// 提供的索引大于最后一个块，相当于最后一个块 <br>
    pub(crate) fn get_prev_index(&self, idx: usize) -> Option<usize> {
        if idx < self.items.len() {
            self.items[idx].thing_prev()
        } else {
            self.items.last()?.thing_prev()
        }
        
    }
    
    /// 通过索引得到当前或下一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    /// 提供的索引大于最后一个块，返回None <br>
    pub(crate) fn get_next_index(&self,idx: usize) -> Option<usize> {
        if idx < self.items.len() {
            self.items[idx].thing_next()
        } else { None }
    }

    
    pub fn first(&self) -> Option<V> {
        Some(self.first_field()?.1.first())
    }
    
    pub fn last(&self) -> Option<V> {
        Some(self.last_field()?.1.last())
    }
    
    
    /// 找到第一个非空块的(键,块引用)，即第一个元素
    pub(crate) fn first_field(&self) -> Option<(usize,&FieldIn<V>)> {
        Some(self.items[self.first_index()?].as_thing())
    }
    
    /// 找到最后一个非空块的(键,块引用)，即最后一个元素
    pub(crate) fn last_field(&self) -> Option<(usize,&FieldIn<V>)> {
        Some(self.items[self.last_index()?].as_thing())
    }
    
    
    /// 找到第一个非空块的索引
    pub(crate) fn first_index(&self) -> Option<usize> {
        self.items.first()?.thing_prev()
    }
    
    /// 找到最后一个非空块的索引
    pub(crate) fn last_index(&self) -> Option<usize> {
        self.items.last()?.thing_prev()
    }
    
}
