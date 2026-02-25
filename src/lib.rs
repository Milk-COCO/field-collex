//! # field-collex
//!
//! 基于**递归分块思想**构造的集合库，致力于给需要有序集合中的大量最值查询的场景提供O(1)方案
//! A Rust collection library based on the **recursive block-based idea**, aimed to providing an O(1) solution for scenarios requiring extensive extremum queries in an ordered set
//!
//! ## 核心 Trait / Core Traits
//!   - `Collexetable<V>`：定义「可被 `FieldCollex` 集合管理的元素」的核心行为（提取数值、比较、去重）
//!     `Collexetable<V>`：Define the core behavior of "elements that can be managed by `FieldCollex`" (extracting values, comparing, deduplicating)；
//!   - `FieldValue`：约束数值类型的核心能力（零值、单位值、数值转换）
//!     `FieldValue`：Core capabilities for constraining numerical types (zero, unit value, conversion)；
//!
//! ## 核心模块 / Core Mods
//!
//! ### 1. `collex`
//! `FieldCollex<E, V>`：
//! - 存入 E ，从此 `Collexetable` Trait 的方法得到 V，根据 V 来顺序排序。 E 需要实现 `Collexetable` 来定义得到 V 的方法。
//!   Store E , and use the method of `Collexetable` Trait to get V, then sort them in order based on V. E needs to impl 'Collexetable' to define the method for getting V.
//! - 内部有一些特殊的计算，因此 V 需要实现 `FieldValue` 来支持这些计算。整数原语类型已实现。
//!   There are some special calculations internally, so V needs to impl 'FieldValue' to support these. The integer primitive type has been implemented.
//!
//! ### 2. `set`
//! `FieldSet<V>`：
//! - 基于 `collex` 模块封装的简易版本，使 `E = V`，省去了 `Collexetable`
//!   A simplified version based on the `collex` module, which makes `E=V` to eliminate `Collexetable`
//! - 但 `FieldValue` 依旧需要
//!   But `FieldValue` still requires
//!
//! ## 核心概念 / Core Ideas
//!
//! ### 1. `Span`
//! 依赖 `span_core` 库的 `Span<V>` 类型，定义集合的数值范围：
//! The 'Span<V>' type, which depends on the 'Span_comore' library, defines the numerical range of the collex:
//! - 有限区间：`[start, end)`（闭开区间，`start <= v < end` 视为有效）；
//!   Finite interval: ` [start, end) ` (closed open interval, ` start<=v<end ` is considered valid);
//! - 无限区间：`[start, +∞)`（闭区间，`start <= v` 视为有效）；
//!   Infinite interval: ` [start,+∞) ` (closed interval, ` start<=v ` is considered valid);
//! - 空区间：`start >= end` 的有限区间视为空，无法用于构造集合。
//!   Empty interval: A finite interval with 'start>=end' is considered empty and cannot be used to construct a collex.
//!
//! ### 2. 分块思想 / Block-based
//! 以 `unit`（块大小）为粒度，将 `Span` 划分为多个块，每个块可存储：
//! Using 'unit' (block size) as the granularity, divide 'Span' into multiple blocks, each of which can store:
//! - 单个数值（`Field::Elem`）
//!   Single value (`Field: Elem`)
//! - 子集合（`Field::Collex`）：块内多个数值时，递归创建子 `FieldCollex` 实现分层存储；
//!   Subcollex (`Field: Collex`): When there are multiple values in a block, recursively create a sub FieldCollex to achieve hierarchical storage;
//! - 空（详见 `RawField`）：若某个块为空，会存储前后一个非空块的索引。
//!   Empty (see `RawField` for details): If a block is empty, the index of the preceding and following non-empty blocks will be stored.
//!
//! ### 3. `FieldValue`
//! 约束数值类型的核心 Trait，需实现以下能力：
//! The core Trait that constrains numerical types, requires the following capabilities to be implemented:
//! - 零值(`num_traits::Zero`)/单位值：`zero()`/`min_positive()`；
//!   zero(`num_traits::Zero`)/unit value
//! - 数值转换：`into_usize()`/`from_usize()`（块索引计算）；
//!   conversion: ` into_usize() `/` From_usize() ` (block index calculation);
//! - 区间计算：`ceil()`（块数量向上取整）。
//!   Interval calculation: ceil() (rounding up to get number of blocks).
//!
//! ## 快速开始 / Let's Go
//!
//! ### 1. 引入依赖 / Import
//! 在 `Cargo.toml` 中添加：
//! ```toml
//! [dependencies]
//! field-collex = "0.1.0"
//! span_core = "0.1.0"
//! num-traits = "0.2"
//! thiserror = "1.0"
//! ```
//!
//! ### 2. 基础使用 / Example
//! ```rust
//! use field_collex::FieldSet;
//! use span_core::Span;
//!
//! // 1. 实现 FieldValue 。省略，u32 已实现。
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 2. 定义区间和块大小
//!     let span = Span::new_finite(0u32, 100u32); // 有限区间 [0, 100)
//!     let unit = 10u32; // 每个块大小为 10
//!
//!     // 3. 构造 FieldSet
//!     let mut set = FieldSet::<u32>::new(span, unit)?;
//!
//!     // 4. 插入数值
//!     set.insert(5u32)?;
//!     set.insert(15u32)?;
//!
//!     // 5. 查询数值
//!     assert!(set.contains(5u32));
//!     assert_eq!(set.find_gt(10u32)?, 15u32);
//!
//!     // 6. 批量插入
//!     let result = set.try_extend(vec![25u32, 35u32, 105u32]);
//!     assert_eq!(result.out_of_span, vec![105u32]); // 105 超出区间
//!
//!     // 7. 删除数值
//!     let removed = set.remove(5u32)?;
//!     assert_eq!(removed, 5u32);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## 核心功能示例
//!
//! ### 1. 集合构造
//! 支持三种构造方式：
//! ```rust
//! use field_collex::FieldSet;
//! use span_core::Span;
//!
//! let span = Span::new_finite(0u32, 100u32);
//! let unit = 10u32;
//!
//! // 空集合构造
//! let set = FieldSet::<u32>::new(span.clone(), unit)?;
//!
//! // 预分配容量构造
//! let set_with_cap = FieldSet::<u32>::with_capacity(span.clone(), unit, 5)?;
//!
//! // 批量元素构造（自动去重、过滤超出区间的值）
//! let set_with_elems = FieldSet::<u32>::with_elements(span, unit, vec![5u32, 15u32, 105u32])?;
//! ```
//!
//! ### 2. 数值查询
//! 支持精准查询、范围查询：
//! ```rust
//! use field_collex::FieldSet;
//! use span_core::Span;
//!
//! let mut set = FieldSet::<u32>::new(Span::new_finite(0u32, 100u32), 10u32)?;
//! set.extend(vec![5u32, 15u32, 25u32]);
//!
//! // 精准包含判断
//! assert!(set.contains(15u32));
//!
//! // 范围查询
//! assert_eq!(set.find_ge(15u32)?, 15u32); // 大于等于
//! assert_eq!(set.find_lt(20u32)?, 15u32); // 小于
//! ```
//!
//! ### 3. 批量操作
//! 支持批量插入（带结果反馈）、批量扩展：
//! ```rust
//! use field_collex::FieldSet;
//! use span_core::Span;
//!
//! let mut set = FieldSet::<u32>::new(Span::new_finite(0u32, 100u32), 10u32)?;
//!
//! // 批量插入（忽略错误）
//! set.extend(vec![5u32, 15u32, 25u32]);
//!
//! // 批量插入（返回超出区间/已存在的数值）
//! let result = set.try_extend(vec![25u32, 35u32, 105u32]);
//! assert_eq!(result.already_exist, vec![25u32]); // 已存在
//! assert_eq!(result.out_of_span, vec![105u32]); // 超出区间
//! ```
//!
//! ## 设计理念
//!
//! ### 1. 分层分块的性能优化
//! - 小粒度数值集合：单个块存储单个数值（`Field::Elem`），减少递归开销；
//! - 大粒度数值集合：块内递归创建子 `FieldCollex`（`Field::Collex`），避免全集合遍历；
//! - 预分配容量：`with_capacity`/`extend` 提前分配内存，减少动态扩容的性能损耗。
//!
//! ### 2. 类型安全与错误处理
//! - 泛型约束：`V` 必须实现 `FieldValue`，保证数值计算的合法性；
//! - 精细化错误：每个操作的错误类型（如 `InsertFieldCollexError::OutOfSpan`/`AlreadyExist`）精准描述问题，便于业务侧处理；
//! - 零开销包装：`SetElem<V>` 使用 `#[repr(transparent)]`，无额外内存开销，类型转换通过裸指针实现。
//!
//! ## 注意事项
//!
//! 1. **泛型约束**：
//!    - `V` 必须实现 `FieldValue`，否则无法编译；
//!    - 若使用自定义数值类型，需手动实现 `FieldValue` 的所有方法。
//!
//! 2. **区间约束**：
//!    - 构造集合时，`unit` 不能为 0，`Span` 不能为空，否则返回对应错误；
//!    - 插入的数值必须在 `Span` 范围内，否则返回 `OutOfSpan` 错误。
//!
//! 3. **内存管理**：
//!    - 无限区间的 `Span` 会导致 `size()` 返回 `None`，集合可无限扩展，但需注意内存占用；
//!    - `is_empty()` 判断的是「是否有有效数值」，而非「内存占用为 0」，空集合仍可能保留预分配的容量。
//!


#![allow(dead_code)]

use num_traits::{NumOps, Zero};

pub mod collex;
pub mod set;

pub use set::FieldSet;
pub use collex::FieldCollex;
pub use collex::Collexetable;

macro_rules! index_of (
    ($target: expr) => {
        $target.sub(*self.span.start()).div(self.unit).into_usize()
    };
    ($this: expr, $target: expr) => {
        $target.sub(*$this.span.start()).div($this.unit).into_usize()
    }
);

pub(crate) use index_of;

pub trait FieldValue: Ord + Copy + NumOps + Zero {
    fn ceil(&self) -> Self;
    fn min_positive() -> Self;
    fn into_usize(self) -> usize;
    fn from_usize(value: usize) -> Self;
}

macro_rules! impl_field_value_for_int {
    ($int: ty) => {
        impl FieldValue for $int {
            fn ceil(&self) -> Self { *self }
            fn min_positive() -> Self { 1 }
            fn into_usize(self) -> usize {
                self as usize
            }
            fn from_usize(value: usize) -> Self {
                value as $int
            }
        }
    };
}

impl_field_value_for_int!(isize);
impl_field_value_for_int!(usize);
impl_field_value_for_int!(u8);
impl_field_value_for_int!(u16);
impl_field_value_for_int!(u32);
impl_field_value_for_int!(u64);
impl_field_value_for_int!(u128);
impl_field_value_for_int!(i8);
impl_field_value_for_int!(i16);
impl_field_value_for_int!(i32);
impl_field_value_for_int!(i64);
impl_field_value_for_int!(i128);

macro_rules! impl_field_value_for_ratio {
    ($int: ty) => {
impl FieldValue for fraction::Ratio<$int>{
    fn ceil(&self) -> Self {
        self.ceil()
    }
    
    fn min_positive() -> Self {
        Self::new(1,<$int>::MAX)
    }
    
    fn into_usize(self) -> usize {
        let (a,b) = self.into_raw();
        (a/b).into_usize()
    }
    
    fn from_usize(value: usize) -> Self {
        Self::from_integer(value as $int)
    }
}
    };
}

impl_field_value_for_ratio!(isize);
impl_field_value_for_ratio!(usize);
impl_field_value_for_ratio!(u8);
impl_field_value_for_ratio!(u16);
impl_field_value_for_ratio!(u32);
impl_field_value_for_ratio!(u64);
impl_field_value_for_ratio!(u128);
impl_field_value_for_ratio!(i8);
impl_field_value_for_ratio!(i16);
impl_field_value_for_ratio!(i32);
impl_field_value_for_ratio!(i64);
impl_field_value_for_ratio!(i128);

pub(crate) trait FieldItem<V> {
    fn first(&self) -> &V;
    fn last(&self) -> &V;
}

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
    
    pub fn as_thing_mut(&mut self) -> (usize, &mut V) {
        match self {
            Self::Thing(t) => (t.0, &mut t.1),
            _ => panic!("Called `RawField::as_thing_mut()` on a not `Thing` value`"),
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
                RawField::Thing(_) => unreachable!(),
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
    /// 当RawField为Thing变体时panic，因为Thing不支持克隆！
    fn clone(&self) -> Self {
        self.partial_clone().expect("Called `RawField::clone` on a `Thing` value")
    }
}

#[derive(Debug)]
pub enum Field<V,C>{
    Elem(V),
    Collex(C)
}

impl<V,C> Field<V,C> {
    /// 取得作为Elem时的内部值
    ///
    /// # Panic
    /// 非Elem时panic
    pub fn into_elem(self) -> V {
        match self{
            Self::Elem(e) => e,
            Self::Collex(_) => panic!("Called `Field::into_elem` on a not `Elem` value")
        }
    }
    
    /// 取得作为Elem时的内部值引用
    ///
    /// # Panic
    /// 非Elem时panic
    pub fn as_elem(&self) -> &V {
        match self{
            Self::Elem(e) => e,
            Self::Collex(_) => panic!("Called `Field::as_elem` on a not `Elem` value")
        }
    }
    
    /// 取得作为Collex时的内部值
    ///
    /// # Panic
    /// 非Collex时panic
    pub fn into_collex(self) -> C {
        match self{
            Self::Collex(c) => c,
            Self::Elem(_) => panic!("Called `Field::into_elem` on a not `Collex` value")
        }
    }
    
    /// 取得作为Collex时的内部值引用
    ///
    /// # Panic
    /// 非Collex时panic
    pub fn as_collex(&self) -> &C {
        match self{
            Self::Collex(c) => c,
            Self::Elem(_) => panic!("Called `Field::as_elem` on a not `Collex` value")
        }
    }
    
    /// 取得作为Collex时的内部值可变引用
    ///
    /// # Panic
    /// 非Collex时panic
    pub fn as_collex_mut(&mut self) -> &mut C {
        match self{
            Self::Collex(c) => c,
            Self::Elem(_) => panic!("Called `Field::as_elem` on a not `Collex` value")
        }
    }
}

impl<E,V> FieldItem<E> for Field<E,FieldCollex<E,V>>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    fn first(&self) -> &E {
        match self{
            Field::Elem(e) => {e}
            Field::Collex(collex) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                collex.first().unwrap()
            }
        }
    }
    
    fn last(&self) -> &E {
        match self{
            Field::Elem(e) => {e},
            Field::Collex(collex) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                collex.last().unwrap()
            }
        }
    }
}

