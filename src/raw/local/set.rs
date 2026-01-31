use span_core::Span;
use num_traits::real::Real;
use std::mem;
use std::ops::*;
use std::vec::Vec;
use thiserror::Error;

/// 一个块。详见 具体容器类型 。
///
/// Thing：本块有元素，存本块索引+Key <br>
/// Prev ：本块无元素，有前一个非空块，存其索引 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者索引 <br>
/// Next ：本块无元素，有后一个非空块，存其索引 <br>
/// Void ：容器完全无任何元素 <br>
///
#[derive(Debug, Clone)]
pub(crate) enum RawField<K>
where K:Copy
{
    Thing((usize,K)),
    Prev (usize),
    Among(usize, usize),
    Next (usize),
    Void,
}

impl<K> RawField<K>
where K:Copy
{
    pub fn as_thing(&self) -> (usize, K) {
        match self {
            Self::Thing(t) => *t,
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
    
    pub fn prev_from(tuple: &(usize,K)) -> Self {
        Self::Prev(tuple.0)
    }
    
    pub fn next_from(tuple: &(usize,K)) -> Self {
        Self::Next(tuple.0)
    }
}

pub(crate) type FindResult<T,I> = Result<T, FindRawFieldSetError<I>>;

#[derive(Error, Debug)]
pub enum FindRawFieldSetError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("目标值超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}

pub(crate) type ReplaceIndexResult<I> = Result<(), ReplaceIndexRawFieldSetError<I>>;

#[derive(Error, Debug)]
pub enum ReplaceIndexRawFieldSetError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("指定的块为空块")]
    EmptyField,
}


pub(crate) type RemoveIndexResult<T> = Result<T, RemoveIndexRawFieldSetError>;

#[derive(Error, Debug)]
pub enum RemoveIndexRawFieldSetError {
    #[error("指定的块已为空块")]
    EmptyField,
}

pub(crate) type TryInsertResult<I> = Result<(), TryInsertRawFieldSetError<I>>;
#[derive(Error, Debug)]
pub enum TryInsertRawFieldSetError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("Key超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("Key对应块已存在元素")]
    AlreadyExists,
}

pub(crate) type InsertResult<K,I> = Result<Option<K>, InsertRawFieldSetError<I>>;
#[derive(Error, Debug)]
pub enum InsertRawFieldSetError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("Key超出了当前RawFieldSet的span范围")]
    OutOfSpan,
    #[error("指定的块正在被借用中")]
    BorrowConflict
}

/// O(1)根据upper_bound或lower_bound查找值
/// 
#[derive(Default, Debug)]
pub struct RawFieldSet<K>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Sized + Real,
{
    span: Span<K>,
    unit: K,
    items: Vec<RawField<K>>,
}

impl<K,IE> RawFieldSet<K>
where
    K: Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize,Error=IE> + Sized + Real,
{
    /// 提供span与unit，构建一个RawFieldSet
    ///
    /// span为Key的范围，unit为每个块的大小，同时也是每个块之间的间隔
    ///
    /// 若unit为0 或 span为空，通过返回Err返还提供的数据
    pub fn new(span: Span<K>, unit: K) -> Result<Self,(Span<K>,K)> {
        if unit.is_zero() || span.is_empty() {
            Err((span, unit))
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
    /// 若unit为0、span为空、转换失败、capacity大于最大块数量，通过返回Err返还提供的数据
    pub fn with_capacity(span: Span<K>, unit: K, capacity: usize) -> Result<Self,(Span<K>,K)> {
        if unit.is_zero() || span.is_empty() ||
            match span.size(){
                Ok(Some(size)) => {
                    capacity >
                        match (size / unit).ceil().try_into() {
                            Ok(v) => {v}
                            _ => return Err((span, unit))
                        }
                },
                Ok(None) => {false}
                _ => {return Err((span, unit));}
            } {
            Err((span, unit))
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
    // /// 若构建失败，返回传入值（`Err(Span<K>, K, Vec<(K,V)>`）。
    // pub fn try_from_inner(span: Span<K>, unit: K, items: Vec<(K,V)>) -> Result<Self,(Span<K>, K, Vec<(K,V)>)> {
    //     if span.size()< unit {
    //         Err((span, unit, items))
    //     } else {
    //         Ok(Self { span, unit, items })
    //     }
    // }
    
    pub fn span(&self) -> &Span<K> {
        &self.span
    }
    
    pub fn unit(&self) -> &K {
        &self.unit
    }
    
    /// 通过索引得到块键
    ///
    /// 若块不为空，返回Some
    pub fn get(&self,idx: usize) -> Option<K> {
        let thing = self.as_thing(idx)?;
        Some(thing.1)
    }
    
    /// 通过索引得到块键
    ///
    /// 若块不为空，返回Some
    ///
    /// # Panics
    /// 越界访问时panic
    pub fn unchecked_get(&self,idx: usize) -> Option<K> {
        match self.items[idx] {
            RawField::Thing(ref t) => Some(t.1),
            _ => None
        }
    }
    
    /// 返回最大块数量
    ///
    /// 若Span是无限区间，返回Ok(None) <br>
    /// 若发生转换错误，返回Err
    pub fn size(&self) -> Result<Option<usize>,IE> {
        // 确保在创建时就不可能为空区间。详见那些构造函数
        Ok(match self.span.size(){
            Ok(Some(size)) => Some((size / self.unit).ceil().try_into() ? ),
            _ => None
        })
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
    
    /// 返回引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查
    pub(crate) fn as_thing(&self, idx: usize) -> Option<(usize, K)> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref v) => Some(*v),
                _ => None
            }
        } else { None }
    }
    
    /// 返回引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查
    pub(crate) fn as_thing_mut(&mut self, idx: usize) -> Option<&mut (usize, K)> {
        if idx < self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref mut v) => Some(v),
                _ => None
            }
        } else { None }
    }
    
    /// 计算指定key对应的块索引，统一错误处理
    #[inline(always)]
    pub(crate) fn idx_of_key(&self, key: K) -> Result<usize, IE> {
        ((key - *self.span.start()) / self.unit).try_into()
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
    
    // 辅助函数：执行插入/替换后的前后更新逻辑
    pub(crate) fn try_insert_in(
        &mut self,
        idx: usize,
        key: K,
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
        
        items[idx] = RawField::Thing((idx,key));
    }
    
    /// 尝试插入键值对
    ///
    /// 插入失败会返回 `TryInsertRawFieldSetError` ，使用 `unwrap` 方法得到传入值 `value`。
    pub fn try_insert(&mut self, key: K) -> TryInsertResult<IE> {
        use TryInsertRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&key) { return Err(OutOfSpan) }
        
        let idx = match self.idx_of_key(key){
            Ok(v) => {v}
            // 需要拿走所有权所以只能这么match
            Err(e) => {return Err(IntoError(e));}
        };
        
        if self.is_thing(idx) {return Err(AlreadyExists)};
        
        self.try_insert_in(
            idx,
            key,
        );
        Ok(())
    }
    
    /// 插入或替换键值对
    ///
    /// 若对应块已有值，新键值将替换原键值，返回Ok(Some(V))包裹原键值。<br>
    /// 若无值，插入新值返回None。
    ///
    pub fn insert(&mut self, key: K) -> InsertResult<K, IE>  {
        use InsertRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&key) { return Err(OutOfSpan) }
        
        let idx = self.idx_of_key(key).map_err(IntoError)?;
        
        if let Some(thing) = self.as_thing_mut(idx){
            // 已存在，则替换并返回其原键
            let key_old = thing.1;
            thing.1 = key;
            Ok(Some(key_old))
        } else {
            // 同 try_insert
            self.try_insert_in(
                idx,
                key,
            );
            
            Ok(None)
        }
    }
    
    
    /// 用索引指定替换块
    ///
    /// 成功则返回其原值
    ///
    /// # Panics
    /// 索引越界时panic
    pub fn unchecked_replace_index(&mut self, idx: usize, key: K) -> ReplaceIndexResult<IE> {
        use ReplaceIndexRawFieldSetError::*;
        
        if let RawField::Thing(ref mut thing) = self.items[idx] {
            thing.1 = key;
            Ok(())
        } else {
            Err(EmptyField)
        }
    }
    
    /// 用索引指定替换块
    ///
    /// 成功则返回其原值
    pub fn replace_index(&mut self, idx: usize, key: K) -> ReplaceIndexResult<IE> {
        use ReplaceIndexRawFieldSetError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField) }
        
        self.unchecked_replace_index(idx,key)
    }
    
    /// 用索引指定清空块，但不进行索引检查
    ///
    /// 若指定块非空，返回内部值。
    ///
    /// # Panics
    /// 索引越界时panic
    pub fn unchecked_remove_index(&mut self, idx: usize) -> RemoveIndexResult<K> {
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
    
    /// 用索引指定清空块。
    ///
    /// 若指定块非空，返回内部值。
    pub fn remove_index(&mut self, idx: usize) -> RemoveIndexResult<K>
    {
        use RemoveIndexRawFieldSetError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField) }
        
        self.unchecked_remove_index(idx)
    }
    
    /// find通用前置检查，返回target对应索引
    pub(crate) fn find_checker(
        &self,
        target: K,
    ) -> FindResult<usize, IE> {
        use FindRawFieldSetError::*;
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan); }
        let items = &self.items;
        let len = items.len();
        if len == 0 { return Err(Empty); }
        
        Ok(self.idx_of_key(target).map_err(IntoError)?.min(len - 1))
    }
    
    /// 通用底层查找核心
    ///
    /// # 参数
    /// - target: 查找目标值
    /// - matcher: 字段匹配器，解耦左右查找的字段匹配逻辑，入参为数据数组+索引，返回匹配的(idx,K)
    /// - cmp: 匹配判定规则  | (当前K, 目标K) -> bool | true = 命中当前项，直接返回V
    /// - lmt: 边界兜底规则  | (当前索引, 数组长度) -> bool | true = 触达边界，返回None
    /// - next: 索引跳转规则 | (当前索引) -> usize | 返回查找目标索引
    ///
    /// 因为查找是O(1)所以暂不使用迭代器
    pub(crate) fn find_in(
        &self,
        target: K,
        matcher: impl Fn(&Self, &RawField<K>) -> FindResult<(usize, K), IE>,
        cmp: impl FnOnce(&K,&K) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<K,IE>
    {
        use FindRawFieldSetError::*;
        
        let idx = self.find_checker(target)?;
        let items = &self.items;
        let len = items.len();
        let current = matcher(self,&items[idx])?;
        
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
        target: K,
        matcher: impl Fn(&Self, &RawField<K>) -> FindResult<(usize, K), IE>,
        cmp: impl FnOnce(&K,&K) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<usize, IE>
    {
        use FindRawFieldSetError::*;
        let idx = self.find_checker(target)?;
        let items = &self.items;
        let len = items.len();
        let current = matcher(self,&items[idx])?;
        
        Ok(if cmp(&current.1, &target) {
            idx
        } else {
            if lmt(idx, len) { return Err(CannotFind); }
            next(idx)
        })
    }
    
    pub(crate) fn matcher_l(this: &Self, field: &RawField<K>) -> FindResult<(usize,K), IE> {
        use FindRawFieldSetError::*;
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
    
    pub(crate) fn matcher_r(this: &Self, field: &RawField<K>) -> FindResult<(usize,K), IE> {
        use FindRawFieldSetError::*;
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
    
    /// 找到最近的小于等于 target 的值
    ///
    pub fn find_le(&self, target: K) -> FindResult<K, IE> {
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
    pub fn find_lt(&self, target: K) -> FindResult<K, IE> {
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
    pub fn find_ge(&self, target: K) -> FindResult<K, IE> {
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
    pub fn find_gt(&self, target: K) -> FindResult<K, IE> {
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
    pub fn find_index_le(&self, target: K) -> FindResult<usize, IE> {
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
    pub fn find_index_lt(&self, target: K) -> FindResult<usize, IE> {
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
    pub fn find_index_ge(&self, target: K) -> FindResult<usize, IE> {
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
    pub fn find_index_gt(&self, target: K) -> FindResult<usize, IE> {
        self.find_index_in(
            target,
            Self::matcher_r,
            |the,tgt| *the > *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
}
