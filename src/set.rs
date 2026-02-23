use span_core::Span;
use num_traits::real::Real;
use std::mem;
use std::ops::{Div, Mul, Range};
use std::vec::Vec;
use thiserror::Error;
use crate::*;

type FieldIn<V> = Field<V,FieldSet<V>>;
type SetField<V> = RawField<Field<V,FieldSet<V>>>;


pub(crate) type NewResult<T,V> = Result<T, NewFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum NewFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit 为0")]
    NonPositiveUnit(Span<V>, V),
}

impl<V> NewFieldSetError<V>{
    pub fn unwrap(self) -> (Span<V>, V) {
        match self {
            Self::NonPositiveUnit(span, unit)
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

impl<V> From<WithCapacityFieldSetError<V>> for WithElementsFieldSetError<V> {
    fn from(value: WithCapacityFieldSetError<V>) -> Self {
        match value {
            WithCapacityFieldSetError::EmptySpan(s, u) => {Self::EmptySpan(s,u)}
            WithCapacityFieldSetError::NonPositiveUnit(s, u) => {Self::NonPositiveUnit(s,u)}
            WithCapacityFieldSetError::OutOfSize(..) => {unreachable!()}
        }
    }
}

impl<V> From<NewFieldSetError<V>> for WithElementsFieldSetError<V> {
    fn from(value: NewFieldSetError<V>) -> Self {
        match value {
            NewFieldSetError::EmptySpan(s, u) => {Self::EmptySpan(s,u)}
            NewFieldSetError::NonPositiveUnit(s, u) => {Self::NonPositiveUnit(s,u)}
        }
    }
}

pub(crate) type WithElementsResult<T,V> = Result<T, WithElementsFieldSetError<V>>;

#[derive(Error, Debug)]
pub enum WithElementsFieldSetError<V>{
    #[error("提供的 span 为空（大小为0）")]
    EmptySpan(Span<V>, V),
    #[error("提供的 unit <= 0")]
    NonPositiveUnit(Span<V>, V),
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


#[derive(Debug)]
pub struct TryExtendResult<V> {
    pub ok: Vec<V>,
    pub out_of_span: Vec<V>,
    pub already_exist: Vec<V>
}


/// 每个块可以存多个内容（通过递归结构实现）
/// 非空块可为单个元素或一个FieldSet，以[`Field`]类型存储。
#[derive(Debug)]
pub struct FieldSet<V>
where
    V: FieldValue,
{
    pub(crate) span: Span<V>,
    pub(crate) unit: V,
    pub(crate) items: Vec<SetField<V>>,
    
}

macro_rules! index_of (
    ($target: expr) => {
        Into::<usize>::into((($target - *self.span.start()) / self.unit))
    };
    ($this: expr, $target: expr) => {
        Into::<usize>::into((($target - *$this.span.start()) / $this.unit))
    }
);

impl<V> FieldSet<V>
where
    V: FieldValue,
{
    const SUB_FACTOR: usize = 64;
    /// 提供span与unit，构建一个FieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<V>, unit: V) -> NewResult<Self,V> {
        use NewFieldSetError::*;
        
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
    
    /// 根据Vec快速构造FieldSet，忽略非法值
    pub fn with_elements(span: Span<V>, unit: V, mut vec: Vec<V>) -> WithElementsResult<Self,V>
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        vec.sort();
        vec.dedup();
        
        let mut new = Self::new(span, unit)?;
        // 第一个非法值的索引
        let first_oob_idx = vec.iter().enumerate().rev().try_for_each(
            |(idx,v)|
            if new.span.contains(&v) {
                Err(idx+1)
            } else {
                Ok(())
            }
        ).err().unwrap_or(0);
        let vec = &vec[0..first_oob_idx];
        if vec.len()==0 {
            // do nothing
        } else if vec.len()==1 {
            let _ = new.insert(vec[0]);
        } else {
            let cap = new.idx_of(vec[first_oob_idx-1]);
            // 预分配
            let items = &mut new.items;
            items.reserve(cap);
            
            // 提前插入第一个的内容
            let mut last_idx = index_of!(new,vec[0]);
            // 存在前置空块（自己为起点(==0)就是不存在）
            if last_idx != 0 {
                items.resize(last_idx ,RawField::Prev(last_idx));
            }
            items.push(RawField::Thing((last_idx,Field::Elem(vec[0]))));
            
            // 遍历插入。0在上面
            let vec = &vec[1..];
            for elem in vec.into_iter() {
                // 与上一个完全相同的情况不存在，已经用了dedup。
                
                let this_idx = index_of!(new,*elem);
                // 若此与上一个处于同一个区间，转为Set
                if this_idx == last_idx {
                    // 确保至少存在一个元素。见上
                    match items.last_mut().unwrap()
                        // 有序集合顺序push导致idx相同时最后一个必定是上一个插入的Thing
                        .as_thing_mut().1{
                        Field::Elem(e) => {
                            let span = Span::Finite({
                                let start = *new.span.start() + new.unit * this_idx;
                                start..start + new.unit
                            });
                            let mut unit = new.unit/Self::SUB_FACTOR;
                            if unit.is_zero() {
                                unit = V::min_positive();
                            }
                            let mut set =
                                FieldSet::with_capacity(
                                    span,
                                    unit,
                                    2
                                ).unwrap_or_else(|err|
                                    panic!("Called `FieldCollex::with_capacity` in `FieldCollex::with_elements` to make a new sub FieldSet, but get a error {err}")
                                );
                            // 此处不用传递，因为二者都必然插入成功：属于span且不相等
                            // TODO：改掉这个insert
                            set.insert(*e).unwrap();
                            set.insert(*elem).unwrap();
                            items[this_idx] = RawField::Thing((this_idx ,Field::Collex(set)))
                        }
                        Field::Collex(set) => {
                            // TODO：改掉这个insert
                            set.insert(*elem).unwrap();
                        }
                    };
                } else { // 与上一个处于不同区间，先填充Among再push自己
                    items.resize(this_idx, RawField::Among(last_idx, this_idx));
                    items.push(RawField::Thing((this_idx, Field::Elem(*elem))))
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
        idx < self.items.len()
    }
    
    /// 查找对应值是否存在
    ///
    pub fn contains(&self, value: V) -> bool {
        let idx = self.idx_of(value);
        if self.contains_idx(idx) {
            match &self.items[idx]
            {
                RawField::Thing((_, k)) =>
                    match k {
                        Field::Elem(e) => { value == *e }
                        Field::Collex(set) => { set.contains(value) }
                    }
                _ => false
            }
        } else { false }
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
        Some(*self.first_field()?.1.first())
    }
    
    pub fn last(&self) -> Option<V> {
        Some(*self.last_field()?.1.last())
    }
    
    pub(crate) fn first_in(&self) -> Option<&V> {
        Some(self.first_field()?.1.first())
    }
    
    pub(crate) fn last_in(&self) -> Option<&V> {
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
        self.items.first()?.thing_next()
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
    
    /// idx块的范围
    pub(crate) fn index_range(&self, idx: usize) -> Range<V>
    where
        V: Mul<usize, Output = V>,
    {
        let start = *self.span.start() + self.unit * idx;
        start..start + self.unit
    }
    
    /// 批量插入元素，忽略错误值。
    pub fn extend(&mut self, mut vec: Vec<V>)
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        vec.sort();
        // 逐个插入
        
        for v in vec {
            match self.insert(v) {
                Err(err) => {
                    match err {
                        InsertFieldSetError::OutOfSpan => {
                            break;
                        }
                        InsertFieldSetError::AlreadyExist => {}
                    }
                }
                _ => {}
            }
        }
    }
    
    /// 批量插入元素，返回插入的情况。
    pub fn try_extend(&mut self, mut vec: Vec<V>) -> TryExtendResult<V>
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        vec.sort();
        // 逐个检查并插入
        
        let mut ok = Vec::new();
        let mut out_of_span = Vec::new();
        let mut already_exist = Vec::new();
        for (idx, v) in vec.iter().enumerate() {
            match self.insert(*v) {
                Ok(_) => {
                    ok.push(*v)
                }
                Err(err) => {
                    match err {
                        InsertFieldSetError::OutOfSpan => {
                            out_of_span.extend_from_slice(&vec[idx..]);
                            break;
                        }
                        InsertFieldSetError::AlreadyExist => {
                            already_exist.push(*v)
                        }
                    }
                }
            }
        }
        TryExtendResult {
            ok, out_of_span, already_exist
        }
    }
    
    pub(crate) fn insert_in_ib(&mut self, idx: usize, value: V) -> (bool, InsertResult)
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        use InsertFieldSetError::*;
        let items = &mut self.items;
        
        let mut need_fill = false;
        // 插入处
        let new = RawField::Thing((
            idx,
            match &items[idx] {
                RawField::Thing(t) => {
                    match &t.1 {
                        Field::Elem(e) => {
                            if *e == value {
                                return (false,Err(AlreadyExist));
                            }
                            let span = Span::Finite({
                                let start = *self.span.start() + self.unit * idx;
                                start..start + self.unit
                            });
                            let mut unit = self.unit/Self::SUB_FACTOR;
                            if unit.is_zero() {
                                unit = V::min_positive();
                            }
                            let mut set =
                                FieldSet::with_capacity(
                                    span,
                                    unit,
                                    2
                                ).unwrap_or_else(|err|
                                    panic!("Called `FieldCollex::with_capacity` in `FieldCollex::insert_in_ib` to make a new sub FieldSet, but get a error {err}")
                                );
                            // 此处不用传递，因为二者都必然插入成功：属于span且不相等
                            set.insert(*e).unwrap();
                            set.insert(value).unwrap();
                            Field::Collex(set)
                        }
                        Field::Collex(_) => {
                            let old = mem::replace(&mut items[idx], RawField::Void);
                            match old {
                                RawField::Thing((_,mut t)) => {
                                    match t {
                                        Field::Collex(ref mut set) => {
                                            let ans = set.insert(value);
                                            match &ans {
                                                Ok(_) => {}
                                                Err(_) => {return (false,ans)}
                                            }
                                            t
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
                    Field::Elem(value)
                }
            }
        ));
        let _ = mem::replace(&mut items[idx], new);
        (need_fill,Ok(()))
    }
    
    pub(crate) fn insert_in_ob(&mut self, idx: usize, value: V) -> InsertResult {
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
    pub fn insert(&mut self, value: V) -> InsertResult
    where
        V: Mul<usize, Output = V>,
        V: Div<usize, Output = V>,
    {
        use InsertFieldSetError::*;
        let span = self.span();
        if !span.contains(&value) { return Err(OutOfSpan) }
        
        let idx = self.idx_of(value);
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
    
    /// 返回是否置空当前块
    pub(crate) fn remove_in(this: &mut Self, idx: usize ) -> bool {
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
                        None => return true,
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
        let _old = mem::replace(&mut this.items[idx], filler);
        // debug_assert!(matches!(old, RawField::Thing((i,FieldIn::Elem(v))) if v == target && i == idx));
        
        false
    }
    
    pub(crate) fn remove_rec(this: &mut Self, target: V, idx: usize) -> (bool, RemoveResult<()>) {
        use RemoveFieldSetError::*;
        let items = &mut this.items;
        (false,
            if let RawField::Thing(ref mut t) = items[idx] {
                match t.1 {
                    Field::Elem(e) => {
                        if e != target {
                            Err(NotExist)
                        } else {
                            return (Self::remove_in(this, idx),Ok(()))
                        }
                    }
                    Field::Collex(ref mut set) => {
                        // 循环直到到最里面那层。
                        // 用idx_of是因为不可能出现超出span或空的情况
                        let sub = Self::remove_rec(set,target,set.idx_of(target));
                        // 错误时 sub.0 是 false，不用额外判断
                        return if sub.0 {
                            (Self::remove_in(this, idx),Ok(()))
                        } else {sub}
                    }
                }
            } else {
                Err(NotExist)
            }
        )
    }
    
    /// 删除对应值
    pub fn remove(&mut self, target: V) -> RemoveResult<()> {
        let idx = self.get_index(target)
            .map_err(Into::<RemoveFieldSetError>::into)?;
        Self::remove_rec(self,target,idx).1
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
        self.items.is_empty() || matches!(self.items[0], RawField::Void)
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
        // is_empty 已检查len>0
        Ok(self.idx_of(target).min(self.len() - 1))
    }
    
}
