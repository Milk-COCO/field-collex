use span_core::Span;
use flag_cell::*;
use num_traits::real::Real;
use num_traits::Zero;
use super::RawField;
use std::mem;
use std::ops::*;
use std::vec::Vec;
use thiserror::Error;

impl<K: Copy,T> Clone for RawField<K,T> {
    /// 克隆内部引用
    /// 
    /// # Panics
    /// 若为 `Thing`，panic 
    fn clone(&self) -> Self {
        match self {
            RawField::Thing(_)
            => panic!("Called `RawField::clone()` on a `Thing` value"),
            RawField::Prev(prev)
            => Self::prev(prev.clone()),
            RawField::Among(prev, next)
            => Self::among(prev.clone(), next.clone()),
            RawField::Next(next)
            => Self::next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
}

type FindResult<T,I> = Result<T, FindRawFieldMapError<I>>;

#[derive(Error, Debug)]
pub enum FindRawFieldMapError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("发生借用冲突")]
    BorrowConflict,
    #[error("目标值超出了当前RawFieldMap的span范围")]
    OutOfSpan,
    #[error("无匹配的数据")]
    CannotFind,
    #[error("当前无数据可查询")]
    Empty,
}


type ReplaceIndexResult<T,I> = Result<T, ReplaceIndexRawFieldMapError<T,I>>;

#[derive(Error, Debug)]
pub enum ReplaceIndexRawFieldMapError<T,I> {
    /// 转换错误（携带失败的 T + 转换错误 I）
    #[error("转换失败，插入的值：{0:?}，错误：{1:?}")]
    IntoError(T, I),
    #[error("指定的块正在被借用中")]
    BorrowConflict(T),
    #[error("指定的块为空块")]
    EmptyField(T),
}

impl<T,I> ReplaceIndexRawFieldMapError<T,I>{
    pub fn unwrap(self) -> T {
        match self {
            ReplaceIndexRawFieldMapError::IntoError(v, _) => {v}
            ReplaceIndexRawFieldMapError::BorrowConflict(v) => {v}
            ReplaceIndexRawFieldMapError::EmptyField(v) => {v}
        }
    }
}


type RemoveIndexResult<T> = Result<T, RemoveIndexRawFieldMapError>;

#[derive(Error, Debug)]
pub enum RemoveIndexRawFieldMapError {
    #[error("指定的块已为空块")]
    EmptyField,
}

type TryInsertResult<T,I> = Result<(), TryInsertRawFieldMapError<T,I>>;
#[derive(Error, Debug)]
pub enum TryInsertRawFieldMapError<T,I> {
    /// 转换错误（携带失败的 T + 转换错误 I）
    #[error("转换失败，插入的值：{0:?}，错误：{1:?}")]
    IntoError(T, I),
    /// Key超出span范围（携带失败的 T）
    #[error("Key超出了当前RawFieldMap的span范围，插入的值：{0:?}")]
    OutOfSpan(T),
    /// Key已存在（携带失败的 T）
    #[error("Key对应块已存在元素，插入的值：{0:?}")]
    AlreadyExists(T),
}

impl<T,I> TryInsertRawFieldMapError<T,I>{
    pub fn unwrap(self) -> T {
        match self {
            TryInsertRawFieldMapError::IntoError(v, _) => {v}
            TryInsertRawFieldMapError::OutOfSpan(v) => {v}
            TryInsertRawFieldMapError::AlreadyExists(v) => {v}
        }
    }
}

type InsertResult<K,V,I> = Result<Option<(K,V)>, InsertRawFieldMapError<V,I>>;
#[derive(Error, Debug)]
pub enum InsertRawFieldMapError<T,I> {
    /// 转换错误（携带失败的 T + 转换错误 I）
    #[error("转换失败，插入的值：{0:?}，错误：{1:?}")]
    IntoError(T, I),
    /// Key超出span范围（携带失败的 T）
    #[error("Key超出了当前RawFieldMap的span范围，插入的值：{0:?}")]
    OutOfSpan(T),
    #[error("指定的块正在被借用中")]
    BorrowConflict(T)
}

impl<T,I> InsertRawFieldMapError<T,I>{
    pub fn unwrap(self) -> T {
        match self {
            InsertRawFieldMapError::IntoError(v, _) => {v}
            InsertRawFieldMapError::OutOfSpan(v) => {v}
            InsertRawFieldMapError::BorrowConflict(v) => {v}
        }
    }
}


/// 可使用Key快速查找值的升序序列。<br>
/// 因本质是分块思想，每个单元为以块，故命名为 `RawFieldMap`
///
/// 仅单线程下使用
///
/// 将一块区域划分成小块，每一块只能存入**一个或零个**元素 <br>
/// 第n个（从0开始）块代表其中的元素处于`[n*unit,(n+1)unit) (n∈N+)`区间 <br>
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
#[derive(Default)]
pub struct RawFieldMap<K,V>
where
    K: Default + Copy + Ord + Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Zero + Real + Sized,
{
    span: Span<K>,
    unit: K,
    // (Thing所有权的索引，Thing的Key，Thing的引用或所有权)
    items: Vec<RawField<K,V>>,
}

impl<K,V,IE> RawFieldMap<K,V>
where
    K: Default + Copy + Ord + Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize,Error=IE> + Zero + Real + Sized,
{
    /// 提供span与unit，构建一个RawFieldMap
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
    
    /// 提供span与unit，构建一个RawFieldMap
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
    
    // /// 尝试从内部数据构建一个 `RawFieldMap`
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
    
    /// 通过索引得到引用
    ///
    /// 若块不为空，返回Some
    ///
    /// 想得到最接近的Key的Value，使用find系列函数
    pub fn get(&self,idx: usize) -> Option<FlagRef<V>> {
        Some(self.as_thing(idx)?.2.flag_borrow())
    }
    
    /// 通过索引得到引用
    ///
    /// 若块不为空，返回Some
    ///
    /// 想得到最接近的Key的Value，使用find系列函数
    ///
    /// # Panics
    /// 越界访问时panic
    pub fn unchecked_get(&self,idx: usize) -> Option<FlagRef<V>> {
        match self.items[idx] {
            RawField::Thing(ref t) => Some(t.2.flag_borrow()),
            _ => None
        }
    }
    
    /// 通过索引得到块键值对(key,FlagRef<V>)
    ///
    /// 若块不为空，返回Some
    pub fn get_key_value(&self,idx: usize) -> Option<(K,FlagRef<V>)> {
        let thing = self.as_thing(idx)?;
        Some((thing.1,thing.2.flag_borrow()))
    }
    
    /// 通过索引得到块键值对(key,FlagRef<V>)
    ///
    /// 若块不为空，返回Some
    ///
    /// # Panics
    /// 越界访问时panic
    pub fn unchecked_get_key_value(&self,idx: usize) -> Option<(K,FlagRef<V>)> {
        match self.items[idx] {
            RawField::Thing(ref t) => Some((t.1,t.2.flag_borrow())),
            _ => None
        }
    }
    
    /// 通过索引得到块键
    ///
    /// 若块不为空，返回Some
    pub fn get_key(&self,idx: usize) -> Option<K> {
        let thing = self.as_thing(idx)?;
        Some(thing.1)
    }
    
    /// 通过索引得到块键
    ///
    /// 若块不为空，返回Some
    ///
    /// # Panics
    /// 越界访问时panic
    pub fn unchecked_get_key(&self,idx: usize) -> Option<K> {
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
        if idx<=self.items.len() {
            matches!(self.items[idx], RawField::Thing(_))
        } else { false }
    }
    
    /// 返回引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查
    fn as_thing(&self, idx: usize) -> Option<&(usize, K, FlagCell<V>)> {
        if idx<=self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref v) => Some(v),
                _ => None
            }
        } else { None }
    }
    
    /// 返回可变引用
    ///
    /// 索引对应块是非空则返回Some，带边界检查
    fn as_thing_mut(&mut self, idx: usize) -> Option<&mut (usize, K, FlagCell<V>)> {
        if idx<=self.items.len() {
            match self.items[idx] {
                RawField::Thing(ref mut v) => Some(v),
                _ => None
            }
        } else { None }
    }
    
    /// 计算指定key对应的块索引，统一错误处理
    #[inline(always)]
    fn idx_of_key(&self, key: K) -> Result<usize, IE> {
        ((key - *self.span.start()) / self.unit).try_into()
    }
    
    fn resize_to_idx(&mut self, idx: usize) {
        if self.items.len() <= idx {
            let fill_field = if self.items.is_empty() {
                RawField::Void
            } else {
                // 上面已作空校验，不会panic
                self.items.last().unwrap().borrow_prev_or_clone()
            };
            self.items.resize_with(
                idx + 1,
                // 已确保不是Thing
                || fill_field.clone()
            );
        }
    }
    
    // 辅助函数：执行插入/替换后的前后更新逻辑
    fn try_insert_in(
        &mut self,
        idx: usize,
        key: K,
        value: V,
    ) {
        // 扩容到目标索引
        self.resize_to_idx(idx);
        
        let items = &mut self.items;
        
        let cell = FlagCell::new(value);
        let fill_field_maker = || (idx, cell.flag_borrow());
        
        let len = items.len();
        
        // 向后更新
        items[idx+1..len].iter_mut()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                let old = mem::replace(v, RawField::Void);
                let new = match old {
                    RawField::Prev(next) | RawField::Among(_, next) => RawField::Among(fill_field_maker(), next),
                    RawField::Next(_) | RawField::Void => RawField::Next(fill_field_maker()),
                    _ => unreachable!()
                };
                let _ = mem::replace(v, new);
            });
        
        // 向前更新
        items[0..idx].iter_mut().rev()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                let old = mem::replace(v, RawField::Void);
                let new = match old {
                    RawField::Next(prev) | RawField::Among(prev, _) => RawField::Among(prev, fill_field_maker()),
                    RawField::Prev(_) | RawField::Void => RawField::Prev(fill_field_maker()),
                    _ => unreachable!()
                };
                let _ = mem::replace(v, new);
            });
        
        items[idx] = RawField::Thing((idx,key,cell));
    }
    
    /// 尝试插入键值对
    ///
    /// 插入失败会返回 `TryInsertRawFieldMapError` ，使用 `unwrap` 方法得到传入值 `value`。
    pub fn try_insert(&mut self, key: K, value: V) -> TryInsertResult<V, IE>
    {
        use TryInsertRawFieldMapError::*;
        let span = &self.span;
        if !span.contains(&key) { return Err(OutOfSpan(value)) }
        
        let idx = match self.idx_of_key(key){
            Ok(v) => {v}
            // 需要拿走所有权所以只能这么match
            Err(e) => {return Err(IntoError(value,e));}
        };
        
        if self.is_thing(idx) {return Err(AlreadyExists(value))};
        
        self.try_insert_in(
            idx,
            key,
            value
        );
        Ok(())
    }
    
    /// 插入键值对
    ///
    /// 若对应块已有值，新键值将替换原键值，返回Ok(Some(V))包裹原键值。<br>
    /// 若无值，插入新值返回None。
    ///
    /// 插入失败会返回 `InsertRawFieldMapError` ，使用 `unwrap` 方法得到传入值 `(key,value)`。
    pub fn insert(&mut self, key: K, value: V) -> InsertResult<K, V, IE>
    {
        use InsertRawFieldMapError::*;
        let span = &self.span;
        if !span.contains(&key) { return Err(OutOfSpan(value)) }
        
        let idx = match self.idx_of_key(key){
            Ok(v) => {v}
            // 需要拿走所有权所以只能这么match
            Err(e) => {return Err(IntoError(value,e));}
        };
        
        if let Some(thing) = self.as_thing_mut(idx){
            // 已存在，则替换并返回其原键值
            match thing.2.try_replace(value) {
                Ok(v) => {
                    let key_old = thing.1;
                    thing.1 = key;
                    Ok(Some((key_old,v)))
                }
                Err(v) => {Err(BorrowConflict(v))}
            }
        } else {
            // 同 try_insert
            self.try_insert_in(
                idx,
                key,
                value
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
    pub fn unchecked_replace_index(&mut self, idx: usize, value: V) -> ReplaceIndexResult<V,IE> {
        use ReplaceIndexRawFieldMapError::*;
        
        if let RawField::Thing(ref mut thing) = self.items[idx] {
            match thing.2.try_replace(value) {
                Ok(v) => {Ok(v)}
                Err(v) => {Err(BorrowConflict(v))}
            }
        } else {
            Err(EmptyField(value))
        }
    }
    
    /// 用索引指定替换块
    ///
    /// 成功则返回其原值
    pub fn replace_index(&mut self, idx: usize, value: V) -> ReplaceIndexResult<V,IE> {
        use ReplaceIndexRawFieldMapError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField(value)) }
        
        self.unchecked_replace_index(idx,value)
    }
    
    /// 用索引指定清空块，但不进行索引检查
    ///
    /// 若指定块非空，返回内部值。
    ///
    /// # Panics
    /// 索引越界时panic
    pub fn unchecked_remove_index(&mut self, idx: usize) -> RemoveIndexResult<V> {
        use RemoveIndexRawFieldMapError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if let RawField::Thing(_) = items[idx] {
            // 根据上一个元素与下一个元素，生成填充元素
            let next =
                if idx == len-1 {
                    None
                } else {
                    match &items[idx + 1] {
                        RawField::Thing(thing) => Some((thing.0,thing.2.flag_borrow())),
                        RawField::Prev(_)
                        | RawField::Void => None,
                        RawField::Among(_, next)
                        | RawField::Next(next) => Some((next.0,next.1.clone())),
                    }
                };
            
            let prev =
                if idx == 0 {
                    None
                } else {
                    match &items[idx - 1] {
                        RawField::Thing(thing) => Some((thing.0,thing.2.flag_borrow())),
                        RawField::Next(_)
                        | RawField::Void => None,
                        RawField::Among(prev, _)
                        | RawField::Prev(prev) => Some((prev.0,prev.1.clone())),
                    }
                };
            
            let new =
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
            // 见上，new不可能为Thing，故不可能panic。下同
            let old = mem::replace(&mut items[idx], new.clone());
            
            // 向前更新
            items[0..idx].iter_mut().rev()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    let new = new.clone();
                    let _ = mem::replace(v, new);
                });
            
            // 向后更新
            items[idx+1..len].iter_mut()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    let new = new.clone();
                    let _ = mem::replace(v, new);
                });
            
            // 刚刚更新的过程中已确保不再存在任何自己的借用，直接unwrap！
            Ok(old.unwrap())
        } else {
            Err(EmptyField)
        }
    }
    
    /// 用索引指定清空块。
    ///
    /// 若指定块非空，返回内部值。
    pub fn remove_index(&mut self, idx: usize) -> RemoveIndexResult<V>
    {
        use RemoveIndexRawFieldMapError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if idx>=len { return Err(EmptyField) }
        
        self.unchecked_remove_index(idx)
    }
    
    /// find通用前置检查，返回target对应索引
    fn find_checker(
        &self,
        target: K,
    ) -> FindResult<usize, IE> {
        use FindRawFieldMapError::*;
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
    /// - matcher: 字段匹配器，解耦左右查找的字段匹配逻辑，入参为数据数组+索引，返回匹配的(K,V)元组引用
    /// - cmp: 匹配判定规则  | (当前K, 目标K) -> bool | true = 命中当前项，直接返回V
    /// - lmt: 边界兜底规则  | (当前索引, 数组长度) -> bool | true = 触达边界，返回None
    /// - next: 索引跳转规则 | (当前索引) -> usize | 返回查找目标索引
    ///
    /// 因为查找是O(1)所以暂不使用迭代器
    fn find_in(
        &self,
        target: K,
        matcher: impl Fn(&Self, &RawField<K, V>) -> FindResult<(usize, K, FlagRef<V>), IE>,
        cmp: impl FnOnce(&K,&K) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<FlagRef<V>, IE>
    {
        use FindRawFieldMapError::*;
        
        let idx = self.find_checker(target)?;
        let items = &self.items;
        let len = items.len();
        let current = matcher(self,&items[idx])?;
        
        Ok(if cmp(&current.1, &target) {
            current.2
        } else {
            if lmt(idx, len) { return Err(CannotFind); }
            let next = matcher(self,&items[next(idx)])?;
            next.2
        })
    }
    
    fn find_index_in(
        &self,
        target: K,
        matcher: impl Fn(&Self, &RawField<K, V>) -> FindResult<(usize, K, FlagRef<V>), IE>,
        cmp: impl FnOnce(&K,&K) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<usize, IE>
    {
        use FindRawFieldMapError::*;
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
    
    fn matcher_l(this: &Self, field: &RawField<K, V>) -> FindResult<(usize,K,FlagRef<V>), IE> {
        use FindRawFieldMapError::*;
        Ok(match field {
            RawField::Thing(thing)
            =>  (thing.0,thing.1,thing.2.flag_borrow()),
            RawField::Prev(fount)
            | RawField::Among(fount, _)
            => {
                let thing = this.items[fount.0].as_thing();
                (thing.0,thing.1,thing.2.flag_borrow())
            }
            RawField::Next(_)
            => return Err(CannotFind),
            RawField::Void
            => return Err(Empty),
        })
    }
    
    fn matcher_r(this: &Self, field: &RawField<K, V>) -> FindResult<(usize,K,FlagRef<V>), IE> {
        use FindRawFieldMapError::*;
        Ok(match field {
            RawField::Thing(thing)
            =>  (thing.0,thing.1,thing.2.flag_borrow()),
            RawField::Prev(_)
            => return Err(CannotFind),
            RawField::Next(next)
            | RawField::Among(_, next)
            => {
                let thing = this.items[next.0].as_thing();
                (thing.0,thing.1,thing.2.flag_borrow())
            }
            RawField::Void
            => return Err(Empty),
        })
    }
    
    /// 找到最近的小于等于 target 的值
    ///
    pub fn find_le(&self, target: K) -> FindResult<FlagRef<V>, IE> {
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
    pub fn find_lt(&self, target: K) -> FindResult<FlagRef<V>, IE> {
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
    pub fn find_ge(&self, target: K) -> FindResult<FlagRef<V>, IE> {
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
    pub fn find_gt(&self, target: K) -> FindResult<FlagRef<V>, IE> {
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
