//! # field-collex
//!
//! 基于**分块槽位思想**构造的集合库，专为需要有序集合中大量最值查询的场景提供 O(1) 方案。
//! A Rust collection library based on the **block-slot idea**, providing O(1) extremum
//! queries for scenarios requiring frequent ordered-set lookups.
//!
//! ## 核心结构 / Core Types
//!
//! | 类型 | 说明 |
//! |------|------|
//! | [`Collex<E, V>`] | 通用有序集合，元素 E 通过 [`Collexetable`] 提取其数值 V 进行排序 |
//! | [`FieldSet<V>`] | `Collex` 的轻量封装，直接存储 `V`，省去手动实现 `Collexetable` |
//!
//! ## 核心 Trait / Core Traits
//!
//! | Trait | 作用 |
//! |-------|------|
//! | [`Collexetable<V>`] | 定义元素提取排序键值 V 的行为 |
//! | [`FieldValue`] | 约束 V 的数值能力（零值、单位值、数值转换） |
//! | [`ConstUnit`] | 为类型提供编译期常量单位值 `UNIT` |
//!
//! ## 已实现 FieldValue 的类型 / Pre-implemented Types
//!
//! 所有整数原语类型（`i8`~`i128`, `u8`~`u128`, `isize`, `usize`）及
//! 对应的 `fraction::Ratio<T>` 均已实现 `FieldValue` 和 `ConstUnit`。
//! All integer primitives and their `fraction::Ratio<T>` counterparts implement `FieldValue` & `ConstUnit`.
//!
//! ## 设计原理 / Design
//!
//! ### 分块槽位 / Block-slot
//! 以 `unit` 为粒度，将非负数值空间划分为等距槽位，每个槽位可存储 0、1 或多个元素：
//! Value space is partitioned into equidistant slots (stride = `unit`), each holding 0, 1, or many elements:
//!
//! - `Nope` — 空槽，存储 prev/next 非空槽指针，实现 O(1) 链式跳转
//! - `One(T)` — 单个元素
//! - `Many(Vec<T>)` — 多个元素，内部以有序 Vec 存储，支持二分查找
//!
//! 通过对任意值做 `value / unit` 即可 O(1) 定位目标槽位，再配合槽内二分查找
//! 和槽间 prev/next 指针跳转，实现高效的范围查询。
//!
//! ## 快速开始 / Quick Start
//!
//! ```ignore
//! cargo add field-collex
//! ```
//!
//! ```ignore
//! use field_collex::{Collex, Collexetable};
//!
//! // 定义自定义元素类型
//! #[derive(Debug, Clone, PartialEq, Eq)]
//! struct Item { id: u32, name: String }
//!
//! impl Collexetable<u32> for Item {
//!     fn collexate(&self) -> u32 { self.id }
//!     fn collexate_ref(&self) -> &u32 { &self.id }
//!     fn collexate_mut(&mut self) -> &mut u32 { &mut self.id }
//! }
//!
//! let mut c = Collex::<Item, u32>::new();
//! c.insert(Item { id: 10, name: "a".into() }).unwrap();
//! c.insert(Item { id: 5,  name: "b".into() }).unwrap();
//!
//! assert_eq!(c.first().unwrap().id, 5);
//! assert_eq!(c.find_ge(&7).unwrap().id, 10);
//! ```

#![allow(dead_code)]

use num_traits::{NumOps, Zero};

pub mod collex;
pub mod set;

pub use collex::{Collex, ModifyError};
pub use set::FieldSet;

// ===================== Collexetable =====================

/// 定义元素可被 [`Collex`] 管理的核心行为。
///
/// 实现此 trait 的类型可以存入 `Collex<E, V>` 中：
/// - `collexate()` / `collexate_ref()` — 提取用于排序的键值 V
/// - `collexate_mut()` — 可变引用（用于 `modify` 系列操作）
///
/// 提供了 `collex_cmp` / `collex_eq` 等默认方法用于比较。
pub trait Collexetable<V> {
    /// 提取排序键值（所有权）
    fn collexate(&self) -> V;
    /// 获取排序键值的不可变引用
    fn collexate_ref(&self) -> &V;
    /// 获取排序键值的可变引用
    fn collexate_mut(&mut self) -> &mut V;

    /// 基于 collexate 值比较两个元素
    fn collex_cmp<O>(&self, other: &O) -> std::cmp::Ordering
    where
        O: Collexetable<V>,
        V: Ord
    {
        self.collexate_ref().cmp(other.collexate_ref())
    }

    /// 基于 collexate 值判断两元素是否相等
    fn collex_eq<O>(&self, other: &O) -> bool
    where
        O: Collexetable<V>,
        V: Eq
    {
        self.collexate_ref().eq(other.collexate_ref())
    }

    /// 基于 collexate 值判断可变引用下的两元素是否相等
    fn collex_mut_eq<O>(&mut self, other: &mut O) -> bool
    where
        O: Collexetable<V>,
        V: Eq
    {
        self.collexate_ref().eq(other.collexate_ref())
    }
}

// ===================== FieldValue =====================

/// 约束数值类型 V 必须支持的核心能力。
///
/// 要求：`Ord + Copy + NumOps + Zero`
///
/// `Collex` 内部依赖此 trait 进行槽位索引计算(`into_usize` / `from_usize`)
/// 和块数量计算(`ceil`)。
pub trait FieldValue: Ord + Copy + NumOps + Zero {
    /// 向上取整（用于计算最大槽位数）
    fn ceil(&self) -> Self;
    /// 转为 usize（用于数组索引）
    fn into_usize(self) -> usize;
    /// 从 usize 转回
    fn from_usize(value: usize) -> Self;
}

// ===================== ConstUnit =====================

/// 为类型提供编译期常量单位值。
///
/// `Collex::new()` 使用 `V::UNIT` 作为初始槽位宽度。
/// 对于整数类型，`UNIT = 1`。
pub trait ConstUnit {
    /// 编译期单位值常量
    const UNIT: Self;
}

// ===================== impl_field_value_for_int =====================

macro_rules! impl_field_value_for_int {
    ($int: ty) => {
        impl FieldValue for $int {
            fn ceil(&self) -> Self { *self }
            fn into_usize(self) -> usize {
                self as usize
            }
            fn from_usize(value: usize) -> Self {
                value as $int
            }
        }
    };
}

macro_rules! impl_const_unit_for_int {
    ($int: ty) => {
        impl ConstUnit for $int {
            const UNIT: Self = 1;
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

impl_const_unit_for_int!(isize);
impl_const_unit_for_int!(usize);
impl_const_unit_for_int!(u8);
impl_const_unit_for_int!(u16);
impl_const_unit_for_int!(u32);
impl_const_unit_for_int!(u64);
impl_const_unit_for_int!(u128);
impl_const_unit_for_int!(i8);
impl_const_unit_for_int!(i16);
impl_const_unit_for_int!(i32);
impl_const_unit_for_int!(i64);
impl_const_unit_for_int!(i128);

// ===================== impl_field_value_for_ratio =====================

macro_rules! impl_field_value_for_ratio {
    ($int: ty) => {
impl FieldValue for fraction::Ratio<$int>{
    fn ceil(&self) -> Self {
        self.ceil()
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
