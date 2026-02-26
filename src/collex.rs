use span_core::Span;
use std::mem;
use std::ops::Range;
use std::vec::Vec;
use thiserror::Error;
use crate::*;

fn dist_cmp<T: FieldValue>(target:T, a:T, b:T) -> std::cmp::Ordering {
    let dist_a = if a > target { a - target } else { target - a };
    let dist_b = if b > target { b - target } else { target - b };
    dist_a.cmp(&dist_b)
}

type FieldIn<E,V> = Field<E,FieldCollex<E,V>>;
type CollexField<E,V> = RawField<Field<E,FieldCollex<E,V>>>;

pub(crate) type NewResult<T,V> = Result<T, NewFieldCollexError<V>>;

#[derive(Error, Debug)]
pub enum NewFieldCollexError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    NonPositiveUnit(Span<V>, V),
}

impl<V> NewFieldCollexError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::NonPositiveUnit(span, unit)
            | Self::EmptySpan(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type WithCapacityResult<T,V> = Result<T, WithCapacityFieldCollexError<V>>;

#[derive(Error, Debug)]
pub enum WithCapacityFieldCollexError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit <= 0")]
    NonPositiveUnit(Span<V>, V),
    #[error("提供的 capacity 超过最大块数量")]
    OutOfSize(Span<V>, V),
}

impl<V> From<WithCapacityFieldCollexError<V>> for WithElementsFieldCollexError<V> {
    fn from(value: WithCapacityFieldCollexError<V>) -> Self {
        match value {
            WithCapacityFieldCollexError::EmptySpan(s, u) => {Self::EmptySpan(s,u)}
            WithCapacityFieldCollexError::NonPositiveUnit(s, u) => {Self::NonPositiveUnit(s,u)}
            WithCapacityFieldCollexError::OutOfSize(..) => {unreachable!()}
        }
    }
}

impl<V> From<NewFieldCollexError<V>> for WithElementsFieldCollexError<V> {
    fn from(value: NewFieldCollexError<V>) -> Self {
        match value {
            NewFieldCollexError::EmptySpan(s, u) => {Self::EmptySpan(s,u)}
            NewFieldCollexError::NonPositiveUnit(s, u) => {Self::NonPositiveUnit(s,u)}
        }
    }
}

pub(crate) type WithElementsResult<T,V> = Result<T, WithElementsFieldCollexError<V>>;

#[derive(Error, Debug)]
pub enum WithElementsFieldCollexError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit <= 0")]
    NonPositiveUnit(Span<V>, V),
}

impl<V> WithCapacityFieldCollexError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::NonPositiveUnit(span, unit)
            | Self::EmptySpan(span, unit)
            | Self::OutOfSize(span, unit)
            => (span, unit)
        }
    }
}


pub(crate) type GetIndexResult<T> = Result<T, GetIndexFieldCollexError>;

#[derive(Error, Debug)]
pub enum GetIndexFieldCollexError {
    #[error("目标值超出了当前FieldCollex的span范围")]
    OutOfSpan,
    #[error("当前无数据可查询")]
    Empty,
}

macro_rules! impl_from_get_index_err {
    ($err: ident) => {
        impl From<GetIndexFieldCollexError> for $err{
            fn from(value: GetIndexFieldCollexError) -> Self {
                match value {
                    GetIndexFieldCollexError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexFieldCollexError::Empty => {Self::Empty}
                }
            }
        }
    };
    ($err: ident, $empty: ident) => {
        impl From<GetIndexFieldCollexError> for $err{
            fn from(value: GetIndexFieldCollexError) -> Self {
                match value {
                    GetIndexFieldCollexError::OutOfSpan => {Self::OutOfSpan}
                    GetIndexFieldCollexError::Empty => {Self::$empty}
                }
            }
        }
    };
}

impl_from_get_index_err!(RemoveFieldCollexError, NotExist);

pub(crate) type RemoveResult<T> = Result<T, RemoveFieldCollexError>;

#[derive(Error, Debug)]
pub enum RemoveFieldCollexError {
    #[error("目标值超出了当前FieldCollex的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    NotExist,
}


pub(crate) type InsertResult<E> = Result<(), InsertFieldCollexError<E>>;
#[derive(Error, Debug)]
pub enum InsertFieldCollexError<E> {
    #[error("提供值超出了当前FieldCollex的span范围")]
    OutOfSpan(E),
    #[error("已存在此元素")]
    AlreadyExist(E)
}

impl<E> InsertFieldCollexError<E> {
    pub fn unwrap(self) -> E {
        match self {
            Self::AlreadyExist(e) => e,
            Self::OutOfSpan(e) => e,
        }
    }
    
    pub fn map<F,N>(self, f: F) -> InsertFieldCollexError<N>
    where
        F: FnOnce(E) -> N
    {
        use InsertFieldCollexError::*;
        match self {
            AlreadyExist(e) => AlreadyExist(f(e)),
            OutOfSpan(e) => OutOfSpan(f(e)),
        }
    }
}


#[derive(Debug)]
pub struct TryExtendResult<V> {
    pub out_of_span: Vec<V>,
    pub already_exist: Vec<V>
}

pub(crate) type ModifyResult<R,E> = Result<R,ModifyFieldCollexError<R,E>>;
#[derive(Error)]
pub enum ModifyFieldCollexError<R,E> {
    #[error("找不到对应元素")]
    CannotFind,
    #[error("刷新元素位置失败")]
    InsertError(InsertFieldCollexError<(R,E)>),
}

pub trait Collexetable<V> {
    fn collexate(&self) -> V;
    fn collexate_ref(&self) -> &V;
    fn collexate_mut(&mut self) -> &mut V;
    
    fn collex_cmp<O>(&self, other: &O) -> std::cmp::Ordering
    where
        O: Collexetable<V>,
        V: Ord
    {
        self.collexate_ref().cmp(other.collexate_ref())
    }
    
    fn collex_eq<O>(&self, other: &O) -> bool
    where
        O: Collexetable<V>,
        V: Eq
    {
        self.collexate_ref().eq(other.collexate_ref())
    }
    
    fn collex_mut_eq<O>(&mut self, other: &mut O) -> bool
    where
        O: Collexetable<V>,
        V: Eq
    {
        self.collexate_ref().eq(other.collexate_ref())
    }
}

/// 每个块可以存多个内容（通过递归结构实现）
/// 非空块可为单个元素或一个FieldCollex，以[`Field`]类型存储。
///
/// 实际存入E，计算时通过Collexetable<V>中的方法得到V，剩余与FieldSet完全一致
#[derive(Debug)]
pub struct FieldCollex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    pub(crate) span: Span<V>,
    pub(crate) unit: V,
    pub(crate) items: Vec<CollexField<E,V>>,
}

impl<E,V> FieldCollex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue
{
    const SUB_FACTOR: usize = 64;
    /// 提供span与unit，构建一个FieldCollex
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<V>, unit: V) -> NewResult<Self,V> {
        use NewFieldCollexError::*;
        
        if unit <= V::zero() {
            Err(NonPositiveUnit(span, unit))
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
    
    /// 提供span与unit，构建一个FieldCollex
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0、span为空、capacity大于最大块数量，通过返回Err返还提供的数据
    pub fn with_capacity(span: Span<V>, unit: V, capacity: usize) -> WithCapacityResult<Self,V> {
        use WithCapacityFieldCollexError::*;
        if unit <= V::zero() {
            Err(NonPositiveUnit(span, unit))
        } else if span.is_empty() {
            Err(EmptySpan(span, unit))
        } else if match span.size(){
            Ok(Some(size)) => {
                capacity > (size / unit).ceil().into_usize()
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
    
    /// 根据Vec快速构造FieldCollex，忽略非法值
    pub fn with_elements(span: Span<V>, unit: V, mut other: Vec<E>) -> WithElementsResult<Self,V> {
        other.sort_by(Collexetable::collex_cmp);
        other.dedup_by(Collexetable::collex_mut_eq);
        
        let mut new = Self::new(span, unit)?;
        // 第一个非法值的索引
        let first_oob_idx = other.iter().enumerate().rev().try_for_each(
            |(idx,e)|
                if new.span.contains(e.collexate_ref()) {
                    Err(idx+1)
                } else {
                    Ok(())
                }
        ).err().unwrap_or(0);
        let vec = &other[0..first_oob_idx];
        if vec.len()==0 {
            // do nothing
        } else if vec.len()==1 {
            let _ = new.insert(other.into_iter().next().unwrap());
        } else {
            let cap = new.idx_of(vec[first_oob_idx-1].collexate_ref()) + 1;
            // 预分配
            let items = &mut new.items;
            items.reserve(cap);
            
            // 提前插入第一个的内容
            let mut last_idx = index_of!(new,vec[0].collexate_ref());
            // 存在前置空块（自己为起点(==0)就是不存在）
            if last_idx != 0 {
                items.resize(last_idx ,RawField::Prev(last_idx));
            }
            let _ = other.split_off(first_oob_idx);
            let mut vec = other.into_iter();
            items.push(RawField::Thing((last_idx,Field::Elem(vec.next().unwrap()))));
            
            // 遍历插入。0在上面
            for elem in vec {
                // 与上一个完全相同的情况不存在，已经用了dedup。
                
                let this_idx = index_of!(new,elem.collexate_ref());
                // 若此与上一个处于同一个区间，转为Collex
                if this_idx == last_idx {
                    // 确保至少存在一个元素。见上
                    match items[this_idx]
                        // 有序集合顺序push导致idx相同时最后一个必定是上一个插入的Thing
                        .as_thing_mut().1
                    {
                        Field::Elem(_) => {
                            let span = Span::Finite({
                                let start = *new.span.start() + new.unit * V::from_usize(this_idx);
                                start..start + new.unit
                            });
                            let mut unit = new.unit/V::from_usize(Self::SUB_FACTOR);
                            if unit.is_zero() {
                                unit = V::min_positive();
                            }
                            let collex =
                                FieldCollex::with_capacity(
                                    span,
                                    unit,
                                    2
                                ).unwrap_or_else(|err|
                                    panic!("Called `FieldCollex::with_capacity` in `FieldCollex::with_elements` to make a new sub FieldSet, but get a error {err}")
                                );
                            
                            let old =
                                match mem::replace(&mut items[this_idx], RawField::Thing((this_idx ,Field::Collex(collex)))).unwrap() {
                                    Field::Elem(e) => e,
                                    Field::Collex(_) => unreachable!(),
                                };
                            let collex =
                                match items[this_idx]
                                    .as_thing_mut().1 {
                                    Field::Elem(_) => unreachable!(),
                                    Field::Collex(collex) => collex,
                                };
                            // TODO：改掉这个insert
                            if let Err(_) = collex.insert(old) {
                                panic!("Called `FieldCollex::insert` in `FieldCollex::with_elements` but get an unexpected error");
                            }
                            if let Err(_) = collex.insert(elem) {
                                panic!("Called `FieldCollex::insert` in `FieldCollex::with_elements` but get an unexpected error");
                            }
                        }
                        Field::Collex(collex) => {
                            // TODO：改掉这个insert
                            if let Err(_) = collex.insert(elem) {
                                panic!("Called `FieldCollex::insert` in `FieldCollex::with_elements` but get an unexpected error");
                            }
                        }
                    };
                } else { // 与上一个处于不同区间，先填充Among再push自己
                    items.resize(this_idx, RawField::Among(last_idx, this_idx));
                    items.push(RawField::Thing((this_idx, Field::Elem(elem))))
                }
                last_idx = this_idx;
            }
        }
        Ok(new)
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
            Ok(Some(size)) => Some((size / self.unit).ceil().into_usize()),
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
    pub(crate) fn idx_of(&self, target: &V) -> usize {
        target.sub(*self.span.start()).div(self.unit).into_usize()
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
    pub(crate) fn expand_to_with(&mut self, new_size: usize, maker: impl Fn() -> CollexField<E,V>) -> bool {
        if self.items.len() < new_size {
            self.items.resize_with(new_size, maker);
            true
        } else { false }
    }
    
    pub(crate) fn expand_to(&mut self, new_size: usize, filler: CollexField<E,V>) -> bool {
        if self.items.len() < new_size {
            self.items.resize(new_size, filler);
            true
        } else { false }
    }
    
    #[inline(always)]
    #[must_use]
    pub(crate) fn contains_idx(&self, idx: usize) -> bool {
        idx < self.items.len()
    }
    
    /// 查找对应是否存在与其collexate()值相同的元素
    ///
    pub fn contains(&self, elem: &E) -> bool {
        let idx = self.idx_of(elem.collexate_ref());
        if self.contains_idx(idx) {
            match &self.items[idx] {
                RawField::Thing((_, k)) =>
                    match k {
                        Field::Elem(e) => { elem.collex_eq(e) }
                        Field::Collex(collex) => { collex.contains(elem) }
                    }
                _ => false
            }
        } else { false }
    }
    
    /// 查找对应值是否存在
    ///
    pub fn contains_value(&self, value: V) -> bool {
        let idx = self.idx_of(&value);
        if self.contains_idx(idx) {
            match &self.items[idx] {
                RawField::Thing((_, k)) =>
                    match k {
                        Field::Elem(e) => { value.eq(e.collexate_ref())}
                        Field::Collex(collex) => { collex.contains_value(value) }
                    }
                _ => false
            }
        } else { false }
    }
    
    /// 通过索引返回块引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查，越界视为None
    pub(crate) fn get_field(&self, idx: usize) -> Option<&FieldIn<E, V>> {
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
    pub(crate) fn get_prev_field(&self, idx: usize) -> Option<(usize,&FieldIn<E, V>)> {
        Some(self.items[self.get_prev_index(idx)?].as_thing())
    }
    
    /// 通过索引得到当前或下一个非空块的(索引,块引用)
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    /// 提供的索引大于最后一个块，返回None <br>
    pub(crate) fn get_next_field(&self,idx: usize) -> Option<(usize,&FieldIn<E, V>)> {
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
    
    
    pub fn first(&self) -> Option<&E> {
        Some(self.first_field()?.1.first())
    }
    
    pub fn last(&self) -> Option<&E> {
        Some(self.last_field()?.1.last())
    }
    
    
    /// 找到第一个非空块的(键,块引用)，即第一个元素
    pub(crate) fn first_field(&self) -> Option<(usize,&FieldIn<E, V>)> {
        Some(self.items[self.first_index()?].as_thing())
    }
    
    /// 找到最后一个非空块的(键,块引用)，即最后一个元素
    pub(crate) fn last_field(&self) -> Option<(usize,&FieldIn<E, V>)> {
        Some(self.items[self.last_index()?].as_thing())
    }
    
    
    /// 找到第一个非空块的索引
    pub(crate) fn first_index(&self) -> Option<usize> {
        self.items.first()?.thing_next()
    }
    
    /// 找到最后一个非空块的索引
    pub(crate) fn last_index(&self) -> Option<usize> {
        self.items.last()?.thing_prev()
    }
    
    /// target是否可置入idx
    pub(crate) fn is_in_index(&self, idx: usize, target: &V) -> bool {
        self.index_range(idx).contains(target)
    }
    
    /// idx块的范围
    pub(crate) fn index_range(&self, idx: usize) -> Range<V> {
        let start = *self.span.start() + self.unit * V::from_usize(idx);
        start..start + self.unit
    }
    
    /// 批量插入元素，忽略错误值。
    pub fn extend(&mut self, mut vec: Vec<E>) {
        vec.sort_by(Collexetable::collex_cmp);
        // 逐个插入
        
        for v in vec {
            match self.insert(v) {
                Err(err) => {
                    match err {
                        InsertFieldCollexError::OutOfSpan(_) => {
                            break;
                        }
                        InsertFieldCollexError::AlreadyExist(_) => {}
                    }
                }
                _ => {}
            }
        }
    }
    
    /// 批量插入元素，返回插入的情况。
    pub fn try_extend(&mut self, mut vec: Vec<E>) -> TryExtendResult<E> {
        vec.sort_by(Collexetable::collex_cmp);
        // 逐个检查并插入
        
        let mut out_of_span: Vec<E> = Vec::new();
        let mut already_exist = Vec::new();
        let mut vec = vec.into_iter();
        while let Some(v) = vec.next() {
            match self.insert(v) {
                Ok(_) => {
                }
                Err(err) => {
                    match err {
                        InsertFieldCollexError::OutOfSpan(e) => {
                            out_of_span.push(e);
                            out_of_span.extend(vec);
                            break;
                        }
                        InsertFieldCollexError::AlreadyExist(e) => {
                            already_exist.push(e);
                        }
                    }
                }
            }
        }
        TryExtendResult {
            out_of_span, already_exist
        }
    }
    
    pub(crate) fn insert_in_ib(&mut self, idx: usize, value: E) -> (bool, InsertResult<E>) {
        use InsertFieldCollexError::*;
        let items = &mut self.items;
        
        let mut need_fill = false;
        // 插入处
            match &items[idx] {
                RawField::Thing(t) => {
                    match &t.1 {
                        Field::Elem(e) => {
                            if e.collex_eq(&value){
                                return (false,Err(AlreadyExist(value)));
                            }
                            let span = Span::Finite({
                                let start = *self.span.start() + self.unit * V::from_usize(idx);
                                start..start + self.unit
                            });
                            let mut unit = self.unit/V::from_usize(Self::SUB_FACTOR);
                            if unit.is_zero() {
                                unit = V::min_positive();
                            }
                            let collex =
                                FieldCollex::with_capacity(
                                    span,
                                    unit,
                                    2
                                ).unwrap_or_else(|err|
                                    panic!("Called `FieldCollex::with_capacity` in `FieldCollex::insert_in_ib` to make a new sub FieldSet, but get a error {err}")
                                );
                            let old =
                                match mem::replace(&mut items[idx], RawField::Thing((idx ,Field::Collex(collex)))).unwrap() {
                                    Field::Elem(e) => e,
                                    Field::Collex(_) => unreachable!(),
                                };
                            let collex =
                                match items[idx]
                                    .as_thing_mut().1 {
                                    Field::Elem(_) => unreachable!(),
                                    Field::Collex(collex) => collex,
                                };
                            // 此处不用传递，因为二者都必然插入成功：属于span且不相等
                            if let Err(_) = collex.insert(old){
                                panic!("Called `FieldCollex::insert` in `FieldCollex::insert_in_ib` but get an unexpected error");
                            }
                            if let Err(_) = collex.insert(value) {
                                panic!("Called `FieldCollex::insert` in `FieldCollex::insert_in_ib` but get an unexpected error");
                            }
                        }
                        Field::Collex(_) => {
                            let old = mem::replace(&mut items[idx], RawField::Void);
                            match old {
                                RawField::Thing((_,mut t)) => {
                                    match t {
                                        Field::Collex(ref mut collex) => {
                                            let ans = collex.insert(value);
                                            match &ans {
                                                Ok(_) => {}
                                                Err(_) => {return (false,ans)}
                                            }
                                            let _ = mem::replace(
                                                &mut items[idx],
                                                RawField::Thing((idx,t))
                                            );
                                        }
                                        _ => unreachable!()
                                    }
                                }
                                _ => unreachable!()
                            }
                        }
                    }
                }
                _ => {
                    need_fill = true;
                    let _ = mem::replace(
                        &mut items[idx],
                        RawField::Thing((idx,Field::Elem(value)))
                    );
                }
            }
        (need_fill,Ok(()))
    }
    
    pub(crate) fn insert_in_ob(&mut self, idx: usize, value: E) -> InsertResult<E> {
        let items = &mut self.items;
        let len = items.len();
        
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
        Ok(())
    }
    
    /// 插入值
    ///
    pub fn insert(&mut self, value: E) -> InsertResult<E> {
        use InsertFieldCollexError::*;
        let span = self.span();
        if !span.contains(value.collexate_ref()) { return Err(OutOfSpan(value)) }
        
        let idx = self.idx_of(value.collexate_ref());
        // 目标索引越界 -> 根据当前最后一个非空块计算前导 -> reserve expand push
        // 目标索引不越界 -> 填充
        let len = self.len();
        // 未越界（这里同时杜绝了len==0的情况）
        if idx < len {
            let (need_fill,ans) = self.insert_in_ib(idx, value);
            if need_fill {
                let items = &mut self.items;
                // 当前为Thing时，并不需要修改其他块，因为他们存储的索引正常指向当前位置，不需要修改
                // 非第一且前一个非Thing
                if idx != 0 && !matches!(items[idx-1], RawField::Thing(_)) {
                    // 计算前导填充物与填充端点
                    let (first_idx, prev) = match items[idx - 1] {
                        RawField::Prev(prev) | RawField::Among(prev, _) => (prev + 1, RawField::Among(prev, idx)),
                        RawField::Next(_) | RawField::Void => (0, RawField::Next(idx)),
                        RawField::Thing(_) => unreachable!()
                    };
                    
                    items[first_idx..idx].fill(prev);
                }
                // 非最后且后一个非Thing
                if idx != len - 1 && !matches!(items[idx+1], RawField::Thing(_)) {
                    // 计算前导填充物与填充端点
                    let (last_idx, next) = match items[idx + 1] {
                        RawField::Next(next) | RawField::Among(_, next) => (next, RawField::Among(idx, next)),
                        RawField::Prev(_) | RawField::Void => (len, RawField::Prev(idx)),
                        RawField::Thing(_) => unreachable!()
                    };
                    
                    items[idx+1..last_idx].fill(next);
                }
            }
            ans
        } else { // 越界
            self.insert_in_ob(idx, value)
        }
    }
    
    /// 返回(是否置空当前块，删除的块)
    pub(crate) fn remove_in(this: &mut Self, idx: usize ) -> (bool, RemoveResult<CollexField<E,V>>) {
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
        
        let filler =
            match next {
                None =>
                    match prev {
                        None => return (true, Ok(mem::replace(&mut this.items[idx], RawField::Void))),
                        Some(prev) => RawField::Prev(prev),
                    },
                Some(next) =>
                    match prev {
                        None => RawField::Next(next),
                        Some(prev) => RawField::Among(prev, next),
                    },
            };
        
        // 向前更新
        this.items[0..idx].iter_mut().rev()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                *v = filler.clone();
            });
        
        // 向后更新
        this.items[idx+1..len].iter_mut()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                *v = filler.clone();
            });
        
        // 更新自己
        let old = mem::replace(&mut this.items[idx], filler);
        
        (false, Ok(old))
    }
    
    /// (需要清空，Result(值))
    pub(crate) fn remove_rec(this: &mut Self, target: V, idx: usize) -> (bool, RemoveResult<E>) {
        use RemoveFieldCollexError::*;
        let items = &mut this.items;
        (false,
         if let RawField::Thing(ref mut t) = items[idx] {
             match &mut t.1 {
                 Field::Elem(e) => {
                     if target.ne(e.collexate_ref()) {
                         Err(NotExist)
                     } else {
                         let ans = Self::remove_in(this, idx);
                         return (ans.0, ans.1.map(|cf| cf.unwrap().into_elem()))
                     }
                 }
                 Field::Collex(collex) => {
                     // 循环直到到最里面那层。
                     // 用idx_of是因为不可能出现超出span或空的情况
                     let sub = Self::remove_rec(collex,target,collex.idx_of(&target));
                     // 错误时 sub.0 是 false，不用额外判断
                     return if sub.0 {
                         (Self::remove_in(this, idx).0, sub.1)
                     } else {sub}
                 }
             }
         } else {
             Err(NotExist)
         }
        )
    }
    
    /// 删除对应值
    pub fn remove(&mut self, target: V) -> RemoveResult<E> {
        let idx = self.get_index(&target)
            .map_err(Into::<RemoveFieldCollexError>::into)?;
        let ans = Self::remove_rec(self,target,idx);
        if ans.0 {
            self.items.clear()
        }
        ans.1
    }
    
    pub fn find_gt(&self, target: V) -> Option<&E> {
        let last_idx = self.len() - 1;
        self.find_in(
            target,
            |idx| idx+1 ,
            |f| f.thing_or_next(),
            |f,v| f.last().collexate_ref().gt(v),
            move |idx| idx == last_idx,
        )
    }
    
    
    pub fn find_ge(&self, target: V) -> Option<&E> {
        let last_idx = self.len() - 1;
        self.find_in(
            target,
            |idx| idx+1 ,
            |f| f.thing_or_next(),
            |f,v| f.last().collexate_ref().ge(v),
            move |idx| idx == last_idx,
        )
    }
    
    pub fn find_lt(&self, target: V) -> Option<&E> {
        self.find_in(
            target,
            |idx| idx-1 ,
            |f| f.thing_or_prev(),
            |f,v| f.first().collexate_ref().lt(v),
            |idx| idx == 0,
        )
    }
    
    
    pub fn find_le(&self, target: V) -> Option<&E> {
        self.find_in(
            target,
            |idx| idx-1 ,
            |f| f.thing_or_prev(),
            |f,v| f.first().collexate_ref().le(v),
            |idx| idx == 0,
        )
    }
    
    /// 找到最近的大于 target 的值
    ///
    pub(crate) fn find_in(
        &self,
        target: V,
        next: fn(usize) -> usize,
        thing_idx: fn(&CollexField<E, V>) -> Option<Result<usize, usize>>,
        cmp: fn(&FieldIn<E, V>, &V) -> bool,
        is_edge: impl Fn(usize) -> bool,
    ) -> Option<&E> {
        let t_idx = self.get_index(&target).ok()?;
        // 上面get_index内已经判空。
        // 结果落在t位 -> t位的最大值(大于)t -> t位已经足够，进入t位
        //           -> t位的最大值不(大于)t -> t+1位必然超过t位，进入下一位
        // 结果落在非t位 -> 必然超过t位，进入此位
        let f_idx = match thing_idx(&self.items[t_idx])? {
            Ok(idx) => {
                if cmp(&self.items[idx].as_thing().1, &target) {
                    idx
                } else {
                    if is_edge(idx) {
                        return None
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
            Field::Elem(e) => {Some(e)}
            Field::Collex(collex) => {
                collex.find_in(
                    target,
                    next,
                    thing_idx,
                    cmp,
                    is_edge
                )}
        }
    }
    
    /// 找到最近的
    ///
    /// 两边距离相等时返回更小的
    pub fn find_closest(&self, target: V) -> Option<&E> {
        use RawField::*;
        use Field::*;
        
        let t_idx = self.get_index(&target).ok()?;
        // 上面get_index内已经判空。
        // t位是thing -> t位属于t位 -> 是Collex则进入
        //                        -> 是Elem返回其
        //           -> t大于最大 -> t+1最小值与t最大值比较
        //           -> t小于最小 -> t-1最大值与t最小值比较
        // t位不是Thing -> 下一个最小值与上一个最大值比较
        
        match &self.items[t_idx] {
            Thing(field) => {
                match &field.1 {
                    Collex(c) => {
                        let first = field.1.first();
                        if target.ge(first.collexate_ref()){
                            let last = field.1.last();
                            if target.le(last.collexate_ref()) {
                                c.find_closest(target)
                            } else { // 大于最大
                                 Some(
                                     self.items.get(t_idx+1)
                                         .map(|v|
                                             self.thing_dist_cmp_get(target, last,
                                                                     v.as_thing().1.first()
                                             )
                                         )
                                         .unwrap_or(last)
                                 )
                            }
                        } else{ // 小于最小
                            Some(
                                self.items.get(t_idx-1)
                                    .map(|v|
                                        self.thing_dist_cmp_get(target,
                                                                v.as_thing().1.last(),
                                                                first
                                        )
                                    )
                                    .unwrap_or(first)
                            )
                        }
                    }
                    Elem(e) => Some(&e),
                }
            } 
            Next(ans) => Some(&self.items[*ans].as_thing().1.first()),
            Prev(ans) => Some(&self.items[*ans].as_thing().1.last()),
            Among(prev,next) => {
                let prev = self.items[*prev].as_thing().1.last();
                let next = self.items[*next].as_thing().1.first();
                Some(self.thing_dist_cmp_get(target, prev, next))
            }
            Void => None,
        }
    }
    
    pub(crate) fn thing_dist_cmp_get<'a>(&'a self, target:V, prev: &'a E, next: &'a E) -> &'a E{
        use std::cmp::Ordering::*;
        match dist_cmp(target, prev.collexate(), next.collexate()){
            Less => {prev}
            // 不可能存储value相同的元素
            Equal => {prev}
            Greater => {next}
        }
    }
    
    /// 判断本容器是否为空
    ///
    /// 为空不意味着内存占用为0。
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() || matches!(self.items[0], RawField::Void)
    }
    
    /// 计算指定值对应的块索引，但是带通用前置检查
    ///
    /// 获取值对应的索引。
    pub(crate) fn get_index(
        &self,
        target: &V,
    ) -> GetIndexResult<usize> {
        use GetIndexFieldCollexError::*;
        if self.is_empty() { return Err(Empty) }
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan); }
        // is_empty 已检查len>0
        Ok(self.idx_of(target).min(self.len() - 1))
    }
    
    pub fn get(&self, value: V) -> Option<&E> {
        let idx = self.idx_of(&value);
        if self.contains_idx(idx) {
            match &self.items[idx] {
                RawField::Thing((_, k)) =>
                    match k {
                        Field::Elem(e) => {
                            if value.eq(e.collexate_ref()) {
                                Some(e)
                            } else {
                                None
                            }
                        }
                        Field::Collex(collex) => { collex.get(value) }
                    }
                _ => None
            }
        } else { None }
    }
    
    /// # Panics
    /// 找不到panic
    pub fn unchecked_get(&self, value: V) -> &E {
        let idx = self.idx_of(&value);
        match &self.items[idx] {
            RawField::Thing((_, k)) =>
                match k {
                    Field::Elem(e) => {
                        if value.eq(e.collexate_ref()) {
                            e
                        } else {
                            panic!("Called `FieldCollex::unchecked_get_mut()` on a value is not eq to where it points an field's value")
                        }
                    }
                    Field::Collex(collex) => { collex.unchecked_get(value) }
                }
            _ => panic!("Called `FieldCollex::unchecked_get()` on a value points an empty field")
        }
    }
    
    
    pub(crate) fn get_mut(&mut self, value: V) -> Option<&mut E> {
        let idx = self.idx_of(&value);
        if self.contains_idx(idx) {
            match &mut self.items[idx] {
                RawField::Thing((_, k)) =>
                    match k {
                        Field::Elem(e) => {
                            if value.eq(e.collexate_ref()) {
                                Some(e)
                            } else {
                                None
                            }
                        }
                        Field::Collex(collex) => { collex.get_mut(value) }
                    }
                _ => None
            }
        } else { None }
    }
    
    pub(crate) fn unchecked_get_mut(&mut self, value: V) -> &mut E {
        let idx = self.idx_of(&value);
        match &mut self.items[idx] {
            RawField::Thing((_, k)) =>
                match k {
                    Field::Elem(e) => {
                        if value.eq(e.collexate_ref()) {
                            e
                        } else {
                            panic!("Called `FieldCollex::unchecked_get_mut()` on a value is not eq to where it points an field's value")
                        }
                    }
                    Field::Collex(collex) => { collex.unchecked_get_mut(value) }
                }
            _ => panic!("Called `FieldCollex::unchecked_get_mut()` on a value points an empty field")
        }
    }
    
    /// 闭包结束后，会根据Value是否发生变化来决定是否更新其在Collex中的位置
    ///
    /// 若更新后的值无法容纳于本Collex，将直接通过错误类型的变体返还
    pub fn modify<F,R>(&mut self, value: V, op: F) -> ModifyResult<R,E>
    where
        F: Fn(&mut E) -> R
    {
        use ModifyFieldCollexError::*;
        
        let elem = self.get_mut(value).ok_or(CannotFind)?;
        let old_v = elem.collexate();
        let result = op(elem);
        if old_v.eq(elem.collexate_ref()) {
            Ok(result)
        } else {
            // TODO: 优化逻辑
            let new_v = elem.collexate();
            *elem.collexate_mut() = old_v;
            let mut new_e = self.remove(old_v).unwrap();
            *new_e.collexate_mut() = new_v;
            match self.insert(new_e) {
                Ok(()) => Ok(result),
                Err(err) => Err(InsertError(err.map(|e| (result,e)))),
            }
        }
    }
    
    /// 闭包结束后，会根据Value是否发生变化来决定是否更新其在Collex中的位置
    ///
    /// # Panics
    /// 若无法找到对应的元素，panic
    ///
    /// 若更新后的值无法容纳于本Collex，将直接Panic
    pub fn unchecked_modify<F,R>(&mut self, value: V, op: F) -> R
    where
        F: Fn(&mut E) -> R
    {
        let elem = self.unchecked_get_mut(value);
        let old_v = elem.collexate();
        let result = op(elem);
        if old_v.eq(elem.collexate_ref()) {
            // do nothing
        } else {
            let new_v = elem.collexate();
            *elem.collexate_mut() = old_v;
            let mut new_e = self.remove(old_v).unwrap();
            *new_e.collexate_mut() = new_v;
            if let Err(_) = self.insert(new_e) {
                panic!("Called `FieldCollex::insert` in `FieldCollex::unchecked_modify` but get an error");
            }
        }
        result
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use span_core::Span;
    
    // ===================== 测试用元素类型（实现Collexetable<u32>） =====================
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    struct TestElem(u32);
    
    impl Collexetable<u32> for TestElem {
        fn collexate(&self) -> u32 {
            self.0
        }
        
        fn collexate_ref(&self) -> &u32 {
            &self.0
        }
        
        fn collexate_mut(&mut self) -> &mut u32 {
            &mut self.0
        }
    }
    
    // ===================== Pub方法测试用例 =====================
    #[test]
    fn test_basic_construction() {
        // 测试：new / with_capacity / span / unit / size / len / capacity / is_empty
        let finite_span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        
        // 1. new方法（正常场景）
        let collex = FieldCollex::<TestElem, u32>::new(finite_span.clone(), unit).unwrap();
        assert_eq!(collex.span(), &finite_span);
        assert_eq!(*collex.unit(), unit);
        assert_eq!(collex.size(), Some(10)); // 100/10=10块
        assert_eq!(collex.len(), 0);
        assert_eq!(collex.capacity(), 0);
        assert!(collex.is_empty());
        
        // 2. new方法（错误场景：unit=0）
        let err_unit_zero = FieldCollex::<TestElem, u32>::new(finite_span.clone(), 0u32).unwrap_err();
        assert!(matches!(err_unit_zero, NewFieldCollexError::NonPositiveUnit(_, 0)));
        
        // 3. new方法（错误场景：空span）
        let empty_span = Span::new_finite(5u32, 3u32); // start >= end 为空
        let err_empty_span = FieldCollex::<TestElem, u32>::new(empty_span, unit).unwrap_err();
        assert!(matches!(err_empty_span, NewFieldCollexError::EmptySpan(_, _)));
        
        // 4. with_capacity方法（正常场景）
        let collex_with_cap = FieldCollex::<TestElem, u32>::with_capacity(finite_span, unit, 5).unwrap();
        assert_eq!(collex_with_cap.capacity(), 5);
        assert!(collex_with_cap.is_empty());
        
        // 5. with_capacity方法（错误场景：capacity超限）
        let err_cap_out = FieldCollex::<TestElem, u32>::with_capacity(Span::new_finite(0u32, 100u32), 10u32, 11).unwrap_err();
        assert!(matches!(err_cap_out, WithCapacityFieldCollexError::OutOfSize(_, _)));
    }
    
    #[test]
    fn test_insert_contains() {
        // 测试：insert / contains / contains_value / first / last
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut collex = FieldCollex::<TestElem, u32>::new(span, unit).unwrap();
        
        // 插入元素
        let elem1 = TestElem(5);
        let elem2 = TestElem(15);
        assert!(collex.insert(elem1).is_ok());
        assert!(collex.insert(elem2).is_ok());
        
        // 验证包含性
        assert!(collex.contains(&elem1));
        assert!(collex.contains_value(5u32));
        assert!(collex.contains(&elem2));
        assert!(!collex.contains(&TestElem(25)));
        assert!(!collex.contains_value(25u32));
        
        // 验证首尾元素
        assert_eq!(collex.first(), Some(&elem1));
        assert_eq!(collex.last(), Some(&elem2));
        
        // 验证非空
        assert!(!collex.is_empty());
    }
    
    #[test]
    fn test_remove() {
        // 测试：remove
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut collex = FieldCollex::<TestElem, u32>::new(span, unit).unwrap();
        
        // 插入后删除
        let elem = TestElem(5);
        collex.insert(elem).unwrap();
        let removed = collex.remove(5u32).unwrap();
        assert_eq!(removed, elem);
        
        // 验证删除后不包含
        assert!(!collex.contains(&elem));
        assert!(!collex.contains_value(5u32));
        assert!(collex.is_empty());
        
        // 错误场景：删除不存在的值
        let err_remove = collex.remove(10u32).unwrap_err();
        assert!(matches!(err_remove, RemoveFieldCollexError::NotExist));
    }
    
    #[test]
    fn test_extend_try_extend() {
        // 测试：extend / try_extend
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut collex = FieldCollex::<TestElem, u32>::new(span, unit).unwrap();
        
        // 1. extend：批量插入
        let elems = vec![TestElem(5), TestElem(15), TestElem(25)];
        collex.extend(elems.clone());
        assert!(collex.contains(&TestElem(5)));
        assert!(collex.contains(&TestElem(15)));
        
        // 2. try_extend：批量插入并返回结果
        let elems2 = vec![TestElem(25), TestElem(35), TestElem(105)]; // 105超出span范围
        let result = collex.try_extend(elems2);
        // 验证：105超出span，25已存在，35插入成功
        assert!(!result.out_of_span.is_empty() && result.out_of_span[0].0 == 105);
        assert!(!result.already_exist.is_empty() && result.already_exist[0].0 == 25);
        assert!(collex.contains(&TestElem(35)));
    }
    
    #[test]
    fn test_find_methods() {
        // 测试：find_gt / find_ge / find_lt / find_le
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut collex = FieldCollex::<TestElem, u32>::new(span, unit).unwrap();
        
        // 插入测试元素
        let elems = [TestElem(5), TestElem(15), TestElem(25)];
        for &e in &elems {
            collex.insert(e).unwrap();
        }
        
        // 测试find_gt（大于）
        let gt = collex.find_gt(10u32).unwrap();
        assert_eq!(*gt, TestElem(15));
        
        // 测试find_ge（大于等于）
        let ge = collex.find_ge(15u32).unwrap();
        assert_eq!(*ge, TestElem(15));
        
        // 测试find_lt（小于）
        let lt = collex.find_lt(20u32).unwrap();
        assert_eq!(*lt, TestElem(15));
        
        // 测试find_le（小于等于）
        let le = collex.find_le(25u32).unwrap();
        assert_eq!(*le, TestElem(25));
        
        // 错误场景：找不到匹配值
        let err_find = collex.find_gt(30u32);
        assert!(matches!(err_find, None));
    }
    
    #[test]
    fn test_with_elements() {
        // 测试：with_elements（批量构造）
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let elems = vec![TestElem(5), TestElem(15), TestElem(25), TestElem(105)]; // 105超出span
        
        // 构造FieldCollex
        let collex = FieldCollex::<TestElem, u32>::with_elements(span, unit, elems).unwrap();
        // 验证：105被忽略，5/15/25插入成功
        assert!(collex.contains(&TestElem(5)));
        assert!(collex.contains(&TestElem(15)));
        assert!(!collex.contains(&TestElem(105)));
        assert_eq!(collex.first(), Some(&TestElem(5)));
        assert_eq!(collex.last(), Some(&TestElem(25)));
    }
}
