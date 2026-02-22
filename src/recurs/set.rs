use span_core::Span;
use num_traits::real::Real;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::{Div, Mul, Range};
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
pub enum RawField<V, IDX = usize> {
    Thing((IDX, V)),
    Prev (IDX),
    Among(IDX, IDX),
    Next (IDX),
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
    
    /// 得到当前或上一个非空块的索引，使用变体区分
    ///
    /// 若块不为空，返回Some(Ok) <br>
    /// 若块为空且有前一个非空块，返回Some(Err) <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    pub fn thing_or_prev(&self) -> Option<Result<usize,usize>> {
        match self {
            RawField::Thing(v) => Some(Ok(v.0)),
            RawField::Prev(prev)
            | RawField::Among(prev,..)
            => Some(Err(*prev)),
            _ => None,
        }
    }
    
    /// 得到当前或下一个非空块的索引，使用变体区分
    ///
    /// 若块不为空，返回Some(Ok) <br>
    /// 若块为空且有后一个非空块，返回Some(Err) <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    pub fn thing_or_next(&self) -> Option<Result<usize,usize>> {
        match self {
            RawField::Thing(v) => Some(Ok(v.0)),
            RawField::Next(next)
            | RawField::Among(_, next)
            => Some(Err(*next)),
            _ => None,
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
    #[error("提供的 unit <= 0")]
    NonPositiveUnit(Span<V>, V),
    #[error("提供的 capacity 超过最大块数量")]
    OutOfSize(Span<V>, V),
}

impl<V> WithCapacityFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::NonPositiveUnit(span, unit)
            | Self::EmptySpan(span, unit)
            | Self::OutOfSize(span, unit)
            => (span, unit)
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
impl_from_get_index_err!(RemoveFieldSetError, CannotFind);


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


pub(crate) type RemoveResult<T> = Result<T, RemoveFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveFieldSetError {
    #[error("目标值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("指定的块已为空块")]
    NotExist,
}


pub(crate) type InsertResult = Result<(), InsertFieldSetError>;
#[derive(Error, Debug)]
pub enum InsertFieldSetError {
    #[error("提供值超出了当前FieldSet的span范围")]
    OutOfSpan,
    #[error("已存在此元素")]
    AlreadyExist
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
        if unit <= V::zero() {
            Err(NonPositiveUnit(span, unit))
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
    
    /// 将内部Vec大小扩大到 idx+1
    ///
    /// 自动填充后继元素
    ///
    /// 返回值意味着是否进行了大小修改： <br>
    /// 当前大小已达标时，没有进行大小修改，返回false
    pub(crate) fn expand_to_idx(&mut self, idx: usize) -> bool {
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
        self.expand_to(idx + 1, filler)
    }
    
    /// 将内部Vec大小扩大到 new_size
    ///
    /// 返回值意味着是否进行了大小修改： <br>
    /// 当前大小已达标时，没有进行大小修改，返回false
    pub(crate) fn expand_to_with(&mut self, new_size: usize, maker: impl Fn() -> SetField<V>) -> bool {
        if self.items.len() < new_size {
            self.items.resize_with(new_size, maker);
            true
        } else { false }
    }
    
    pub(crate) fn expand_to(&mut self, new_size: usize, filler: SetField<V>) -> bool {
        if self.items.len() < new_size {
            self.items.resize(new_size, filler);
            true
        } else { false }
    }
    
    #[inline(always)]
    #[must_use]
    pub(crate) fn contains_idx(&self, idx: usize) -> bool {
        self.items.len() <= idx
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
    
    /// target是否可置入idx
    pub(crate) fn is_in_index(&self, idx: usize, target: &V) -> bool
    where
        V: Mul<usize, Output = V>,
    {
        self.index_range(idx).contains(target)
    }
    
    /// target是否可置入idx
    pub(crate) fn index_range(&self, idx: usize) -> Range<V>
    where
        V: Mul<usize, Output = V>,
    {
        self.unit*idx..self.unit*(idx+1)
    }
    
    /// 插入值
    ///
    pub fn insert(&mut self, value: V) -> InsertResult
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        use InsertFieldSetError::*;
        let span = self.span();
        if !span.contains(&value) { return Err(OutOfSpan) }
        
        let idx = self.idx_of(value);
        // 目标索引越界 -> 根据当前最后一个元素计算前导，后导为Prev(idx) -> reserve expand push
        // 目标索引不越界 -> 填充
        let items = &mut self.items;
        let len = items.len();
        // 未越界（这里同时杜绝了len==0的情况）
        if idx < len {
            // 非第一且前一个非Thing
            if idx != 0 && !matches!(items[idx-1], RawField::Thing(_)) {
                // 计算前导填充物与填充端点
                let (first_idx,prev) = match items[idx-1] {
                    RawField::Prev(prev) | RawField::Among(prev, _) => (prev+1, RawField::Among(prev, idx)),
                    RawField::Next(_) | RawField::Void => (0,RawField::Next(idx)),
                    RawField::Thing(_) => unreachable!()
                };
                
                // 得到填入范围
                let mut last_idx = MaybeUninit::uninit();
                for i in (0..idx).rev() {
                    if matches!(items[i], RawField::Thing(_)) {
                        last_idx.write(i+1);
                        break
                    }
                }
                
                // SAFETY：for循环不可能运行0次因为idx>0
                unsafe { items[last_idx.assume_init()..idx].fill(prev); }
            }
            // 非最后
            if idx != len-1 {
                // 计算后导填充物
                let next = match items[idx] {
                    RawField::Next(prev) | RawField::Among(prev, _) => RawField::Among(prev, idx),
                    RawField::Prev(_) | RawField::Void => RawField::Prev(idx),
                    _ => unreachable!()
                items[first_idx..idx].fill(prev);
            }
            // 非最后且后一个非Thing
            if idx != len - 1 && !matches!(items[idx+1], RawField::Thing(_)) {
                // 计算前导填充物与填充端点
                let (last_idx,next) =  match items[idx+1] {
                    RawField::Next(next) | RawField::Among(_, next) => (next,RawField::Among(idx, next)),
                    RawField::Prev(_) | RawField::Void => (len,RawField::Prev(idx)),
                    RawField::Thing(_) => unreachable!()
                };
                
                items[idx+1..last_idx].fill(next);
            }
            // 插入处
            let old = mem::replace(&mut items[idx], RawField::Void) ;
            let new = RawField::Thing((
                idx,
                match old {
                    RawField::Thing(mut t) => {
                        match t.1 {
                            Field::Elem(e) => {
                                if e == value {
                                    return Err(AlreadyExist);
                                }
                                let span = Span::Finite(self.unit*idx..self.unit*(idx+1));
                                let mut set =
                                    match FieldSet::with_capacity(
                                        span,
                                        self.unit/64,
                                        2
                                    ){
                                        Ok(s) => s,
                                        // TODO：更灵活的错误处理
                                        // 因为不能直接unwrap(要V:Debug，显然不必要)所以显式匹配
                                        Err(err) => {
                                            panic!("Called `Field::with_capacity` in `Field::insert` to make a new sub FieldSet, but get a error {err}");
                                        }
                                    }
                                ;
                                // 此处不用传递，因为二者都必然插入成功：属于span且不相等
                                set.insert(e).unwrap();
                                set.insert(value).unwrap();
                                Field::Collex(set)
                            }
                            Field::Collex(ref mut set) => {
                                set.insert(value)?;
                                t.1
                            }
                        }
                    }
                    _ => {
                        Field::Elem(value)
                    }
                }
            ));
            let _ = mem::replace(&mut items[idx], new);
            
        } else { // 越界
            // 修改未越界部分
            let prev =
                if len != 0{
                    match &items[len - 1]{
                        RawField::Thing(t) => {
                            debug_assert_eq!(t.0, len-1);
                            RawField::Among(t.0,idx)
                        }
                        _ => {
                            // 计算前导填充物与填充端点
                            let (first_idx,prev) = match items[len - 1] {
                                RawField::Prev(prev) | RawField::Among(prev, _) => (prev+1, RawField::Among(prev, idx)),
                                RawField::Void => (0,RawField::Next(idx)),
                                RawField::Next(_) | RawField::Thing(_) => unreachable!()
                            };
                            
                            items[first_idx..len].fill(prev.clone());
                            prev
                        }
                    }
                } else {
                    RawField::Next(idx)
                }
            ;
            
            // 补充越界部分
            // reserve expand push
            let need_cap  = idx + 1 - len;
            items.reserve(need_cap);
            items.resize(idx, prev);
            items.push(RawField::Thing((idx, Field::Elem(value))));
        }
        Ok(())
    }
    
    pub(crate) fn remove_rec(this: &mut Self, target: V, idx: usize) -> RemoveResult<()> {
        use RemoveFieldSetError::*;
        let items = &mut this.items;
        if let RawField::Thing(ref mut t) = items[idx] {
            match t.1 {
                Field::Elem(e) => {
                    if e != target {
                        Err(NotExist)
                    } else {
                        // 删除的逻辑
                        let len = this.items.len();
                        // 根据上一个元素与下一个元素，生成填充元素
                        let next =
                            if idx == len-1 {
                                None
                            } else {
                                match &this.items[idx + 1] {
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
                                match &this.items[idx - 1] {
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
                        let old = mem::replace(&mut this.items[idx], maker());
                        debug_assert!(matches!(old, RawField::Thing((i,FieldIn::Elem(v))) if v == target && i == idx));
                        
                        // 向前更新
                        this.items[0..idx].iter_mut().rev()
                            .take_while(|v| !matches!(v, RawField::Thing(_)) )
                            .for_each(|v| {
                                *v = maker();
                            });
                        
                        // 向后更新
                        this.items[idx+1..len].iter_mut()
                            .take_while(|v| !matches!(v, RawField::Thing(_)) )
                            .for_each(|v| {
                                *v = maker();
                            });
                        
                        Ok(())
                    }
                }
                Field::Collex(ref mut set) => {
                    // 循环直到到最里面那层。
                    Self::remove_rec(set,target,set.idx_of(target))
                }
            }
        } else {
            Err(NotExist)
        }
    }
    
    /// 删除对应值
    pub fn remove(&mut self, target: V) -> RemoveResult<()> {
        let idx = self.get_index(target)
            .map_err(Into::<RemoveFieldSetError>::into)?;
        Self::remove_rec(self,target,idx)
    }
    
    pub fn find_gt(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            |idx| idx+1 ,
            |f| f.thing_or_next(),
            |f,v| f.last().gt(v)
        )
    }
    
    
    pub fn find_ge(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            |idx| idx+1 ,
            |f| f.thing_or_next(),
            |f,v| f.last().ge(v)
        )
    }
    
    pub fn find_lt(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            |idx| idx-1 ,
            |f| f.thing_or_prev(),
            |f,v| f.first().lt(v)
        )
    }
    
    
    pub fn find_le(&self, target: V) -> FindResult<V> {
        self.find_in(
            target,
            |idx| idx-1 ,
            |f| f.thing_or_prev(),
            |f,v| f.first().le(v)
        )
    }
    
    /// 找到最近的大于 target 的值
    ///
    pub(crate) fn find_in(
        &self,
        target: V,
        next: fn(usize) -> usize,
        getter: fn(&SetField<V>) -> Option<Result<usize, usize>>,
        cmp: fn(&FieldIn<V>, &V) -> bool
    ) -> FindResult<V> {
        use FindFieldSetError::*;
        
        let t_idx = self.get_index(target)
            .map_err(Into::<FindFieldSetError>::into)?;
        // 上面get_index内已经判空。
        // 结果落在t位 -> t位的最大值(大于)t -> t位已经足够，进入t位
        //           -> t位的最大值不(大于)t -> t+1位必然超过t位，进入下一位
        // 结果落在非t位 -> 必然超过t位，进入此位
        let f_idx = match getter(&self.items[t_idx]).ok_or(CannotFind)? {
            Ok(idx) => {
                if cmp(&self.items[idx].as_thing().1, &target) {
                    idx
                } else {
                    if self.is_edge(idx) {
                        return Err(CannotFind)
                    } else {
                        next(idx)
                    }
                }
            }
            Err(idx) => {
                idx
            }
        };
        
        // 必然是thing
        match self.items[f_idx].as_thing().1 {
            Field::Elem(e) => {Ok(*e)}
            Field::Collex(set) => {
                set.find_in(
                    target,
                    next,
                    getter,
                    cmp
                )}
        }
    }
    
    /// 判断本容器是否为空
    ///
    /// 为空不意味着内存占用为0。
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && matches!(self.items[0], RawField::Void)
    }
    
    pub(crate) fn is_edge(&self, idx: usize) -> bool {
        idx == 0 || idx == self.len()-1
    }
    
    /// 计算指定值对应的块索引，但是带通用前置检查
    ///
    /// 获取值对应的索引。
    pub(crate) fn get_index(
        &self,
        target: V,
    ) -> GetIndexResult<usize> {
        use GetIndexFieldSetError::*;
        if self.is_empty() { return Err(Empty) }
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan); }
        
        Ok(self.idx_of(target).min(self.len() - 1))
    }
    
}
