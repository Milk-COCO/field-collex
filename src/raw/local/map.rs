use span_core::Span;
use flag_cell::*;
use num_traits::real::Real;
use num_traits::Zero;
use std::cell::Ref;
use std::mem;
use std::ops::*;
use std::vec::Vec;
use thiserror::Error;

/// 一个块。详见 [`RawFieldMap`] 。
///
/// Thing：本块有元素 <br>
/// Prev ：本块无元素，有前一个非空块，存其引用 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者引用 <br>
/// Next ：本块无元素，有后一个非空块，存其引用 <br>
/// Void ：容器完全无任何元素 <br>
///
/// Hint：外部 **never** 提取内部类型。随意获取引用可能导致触发panic，或导致Map无法正常工作。
#[derive(Debug)]
enum RawField<T> {
    Thing(FlagCell<T>),
    Prev (FlagRef<T>),
    Among(FlagRef<T>, FlagRef<T>),
    Next (FlagRef<T>),
    Void,
}

impl<T> RawField<T> {
    pub fn partial_clone(&self) -> Option<RawField<T>> {
        match self {
            RawField::Thing(_)
            => None,
            RawField::Prev(prev)
            => Some(RawField::Prev(prev.clone())),
            RawField::Among(prev, next)
            => Some(RawField::Among(prev.clone(), next.clone())),
            RawField::Next(next)
            => Some(RawField::Next(next.clone())),
            RawField::Void
            => Some(RawField::Void),
        }
    }
    
    
    pub fn borrow_prev_or_clone(&self) -> RawField<T> {
        match self {
            RawField::Thing(thing)
            => RawField::Prev(thing.flag_borrow()),
            RawField::Prev(prev)
            => RawField::Prev(prev.clone()),
            RawField::Among(prev, next)
            => RawField::Among(prev.clone(), next.clone()),
            RawField::Next(next)
            => RawField::Next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
    
    pub fn borrow_next_or_clone(&self) -> RawField<T> {
        match self {
            RawField::Thing(thing)
            => RawField::Next(thing.flag_borrow()),
            RawField::Prev(prev)
            => RawField::Prev(prev.clone()),
            RawField::Among(prev, next)
            => RawField::Among(prev.clone(), next.clone()),
            RawField::Next(next)
            => RawField::Next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
    
    /// 解包得到内部值
    ///
    /// 若非Thing，或正在被借用，返回Err返还self
    #[allow(dead_code)]
    pub fn try_unwrap(self) -> Result<T,Self> {
        if let RawField::Thing(t) = self {
            t.try_unwrap().map_err(|t| RawField::Thing(t))
        } else {
            Err(self)
        }
    }
    
    
    /// 解包得到内部值
    ///
    /// 若非Thing，或正在被借用，panic
    pub fn unwrap(self) -> T {
        if let RawField::Thing(t) = self {
            t.unwrap()
        } else {
            panic!("called `RawField::unwrap()` on a not `Thing` value")
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

type RemoveResult<T,I> = Result<T, RemoveRawFieldMapError<I>>;

#[derive(Error, Debug)]
pub enum RemoveRawFieldMapError<I> {
    #[error(transparent)]
    IntoError(I),
    #[error("指定的块已为空块")]
    AlreadyEmpty,
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
    items: Vec<RawField<(K, V)>>,
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
                self.items.last().unwrap().borrow_prev_or_clone()
            };
            self.items.resize_with(
                idx + 1,
                // 防御性。上面已经确保不会是Thing了，但还是unwrap_or
                || fill_field.partial_clone().unwrap_or(RawField::Void)
            );
        }
    }
    
    // 辅助函数：执行插入/替换后的前后更新逻辑
    fn insert_update_prev_next(
        &mut self,
        idx: usize,
        flag_ref: FlagRef<(K, V)>,
    ) {
        let items = &mut self.items;
        let len = items.len();
        
        // 向后更新
        items[idx+1..len].iter_mut()
            .take_while(|v| !matches!(v, RawField::Thing(_)) )
            .for_each(|v| {
                let old = mem::replace(v, RawField::Void);
                let new = match old {
                    RawField::Prev(next) | RawField::Among(_, next) => RawField::Among(flag_ref.clone(), next),
                    RawField::Next(_) | RawField::Void => RawField::Next(flag_ref.clone()),
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
                    RawField::Next(prev) | RawField::Among(prev, _) => RawField::Among(prev, flag_ref.clone()),
                    RawField::Prev(_) | RawField::Void => RawField::Prev(flag_ref.clone()),
                    _ => unreachable!()
                };
                let _ = mem::replace(v, new);
            });
    }
    
    /// 尝试插入键值对
    ///
    /// 插入失败会返回 `TryInsertRawFieldMapError` ，使用 `unwrap` 方法得到传入值 `(key,value)`。
    pub fn try_insert(&mut self, key: K, value: V) -> TryInsertResult<(K, V), IE>
    {
        use TryInsertRawFieldMapError::*;
        let tuple = (key,value);
        let span = &self.span;
        if !span.contains(&key) { return Err(OutOfSpan(tuple)) }
        // 计算目标索引并防越界
        let idx =
                TryInto::<usize>::try_into(
                    (key - *span.start())/self.unit
                ).map_err(IntoError)?;
        
        let items = &mut self.items;
        
        if let RawField::Thing(_) = items[idx] {return Err(AlreadyExists(tuple))};
        
        // 扩容到目标索引
        self.resize_to_idx(idx);
        
        let cell = FlagCell::new(tuple);
        let flag_ref = cell.flag_borrow();
        items[idx] = RawField::Thing(cell);
        self.insert_update_prev_next(
            idx,
            flag_ref
        );
        Ok(())
    }
    
    
    /// 清空指定块。
    ///
    /// 若指定块非空，返回内部值。
    pub fn remove(&mut self, idx: usize) -> RemoveResult<V, IE>
    {
        use RemoveRawFieldMapError::*;
        
        let items = &mut self.items;
        let len = items.len();
        
        if(idx>=len) { return Err(AlreadyEmpty) }
        
        if let RawField::Thing(_) = items[idx] {
            // 根据上一个元素与下一个元素，生成填充元素
            let next =
                if idx == len-1 {
                    None
                } else {
                    match &items[idx + 1] {
                        RawField::Thing(cell) => Some(cell.flag_borrow()),
                        RawField::Prev(_)
                        | RawField::Void => None,
                        RawField::Among(_, next)
                        | RawField::Next(next) => Some(next.clone()),
                    }
                };
            
            let prev =
                if idx == 0 {
                    None
                } else {
                    match &items[idx - 1] {
                        RawField::Thing(cell) => Some(cell.flag_borrow()),
                        RawField::Next(_)
                        | RawField::Void => None,
                        RawField::Among(prev, _)
                        | RawField::Prev(prev) => Some(prev.clone()),
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
            let old = mem::replace(&mut items[idx], new.partial_clone().unwrap());
            
            // 向前更新
            items[0..idx].iter_mut().rev()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    let new = new.partial_clone().unwrap();
                    let _ = mem::replace(v, new);
                });
            
            // 向后更新
            items[idx+1..len].iter_mut()
                .take_while(|v| !matches!(v, RawField::Thing(_)) )
                .for_each(|v| {
                    let new = new.partial_clone().unwrap();
                    let _ = mem::replace(v, new);
                });
            
            // 刚刚更新的过程中已确保不再存在任何自己的借用，直接unwrap！
            Ok(old.unwrap().1)
        } else {
            Err(AlreadyEmpty)
        }
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
    fn find_in(
        &self,
        target: K,
        matcher: impl Fn(&RawField<(K, V)>) -> FindResult<Ref<'_, (K, V)>, IE>,
        cmp: impl FnOnce(&K,&K) -> bool,
        lmt: impl FnOnce(usize,usize) -> bool,
        next: impl FnOnce(usize) -> usize,
    ) -> FindResult<Ref<'_, V>, IE> {
        use FindRawFieldMapError::*;
        let span = &self.span;
        if !span.contains(&target) { return Err(OutOfSpan); }
        let items = &self.items;
        let len = items.len();
        if len == 0 { return Err(Empty); }
        
        let idx = self.idx_of_key(target).map_err(IntoError)?.min(len - 1);
        
        let current = matcher(&items[idx])?;
        
        Ok(if cmp(&current.0, &target) {
            Ref::map(current, |t| &t.1)
        } else {
            if lmt(idx, len) { return Err(CannotFind); }
            Ref::map(matcher(&items[next(idx)])?, |t| &t.1)
        })
    }
    
    fn matcher_l(field: &RawField<(K, V)>) -> FindResult<Ref<'_, (K, V)>, IE> {
        use FindRawFieldMapError::*;
        Ok(match field {
            RawField::Thing(thing)
            => thing.try_borrow().ok_or(BorrowConflict)?,
            RawField::Prev(fount)
            | RawField::Among(fount, _)
            => fount.try_borrow().into_option().ok_or(BorrowConflict)?,
            RawField::Next(_)
            => return Err(CannotFind),
            RawField::Void
            => return Err(Empty),
        })
    }
    
    fn matcher_r(field: &RawField<(K, V)>) -> FindResult<Ref<'_, (K, V)>, IE> {
        use FindRawFieldMapError::*;
        Ok(match field {
            RawField::Thing(thing)
            => thing.try_borrow().ok_or(BorrowConflict)?,
            RawField::Prev(_)
            => return Err(CannotFind),
            RawField::Next(next)
            | RawField::Among(_, next)
            => next.try_borrow().into_option().ok_or(BorrowConflict)?,
            RawField::Void
            => return Err(Empty),
        })
    }
    
    /// 找到最近的小于等于 target 的值
    ///
    pub fn find_le(&self, target: K) -> FindResult<Ref<'_, V>, IE> {
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
    pub fn find_lt(&self, target: K) -> FindResult<Ref<'_, V>, IE> {
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
    pub fn find_ge(&self, target: K) -> FindResult<Ref<'_, V>, IE> {
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
    pub fn find_gt(&self, target: K) -> FindResult<Ref<'_, V>, IE> {
        self.find_in(
            target,
            Self::matcher_r,
            |the,tgt| *the > *tgt,
            |idx,len| idx == len-1,
            |idx| idx+1
        )
    }
}
