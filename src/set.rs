use crate::{Collexetable, ConstUnit, FieldValue};
use crate::collex::Collex;

// ===================== SetElem =====================

/// [`FieldSet`] 内部使用的透明包装。
///
/// 使 `V` 自身实现 [`Collexetable<V>`]，从而可存入 [`Collex`]。
/// `#[repr(transparent)]` 保证内存布局与 `V` 完全一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct SetElem<V: FieldValue>(V);

impl<V: FieldValue> Collexetable<V> for SetElem<V> {
    fn collexate(&self) -> V {
        self.0
    }
    fn collexate_ref(&self) -> &V {
        &self.0
    }
    fn collexate_mut(&mut self) -> &mut V {
        &mut self.0
    }
}

// ===================== FieldSet =====================

/// 有序数值集合，内部基于 [`Collex`] 的轻量封装。
///
/// 省去手动实现 [`Collexetable`]，直接存储 `V` 类型值。
/// 支持自动去重、负数过滤、多种范围查找和最邻近查找。
///
/// ## 示例
/// ```ignore
/// use field_collex::FieldSet;
///
/// let mut set = FieldSet::<u32>::new();
/// set.insert(5).unwrap();
/// set.insert(15).unwrap();
/// set.insert(25).unwrap();
///
/// assert!(set.contains(15));
/// assert_eq!(set.find_gt(10), Some(15));
/// assert_eq!(set.find_closest(12), Some(15));
/// assert_eq!(set.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct FieldSet<V: FieldValue + ConstUnit> {
    collex: Collex<SetElem<V>, V>,
}

impl<V: FieldValue + ConstUnit> FieldSet<V> {
    /// 构造一个空集合。
    pub fn new() -> Self {
        Self {
            collex: Collex::new(),
        }
    }

    /// 从 `Vec` 批量构造，自动去重并忽略负数。
    pub fn with_elements(other: Vec<V>) -> Self {
        let mut set = Self::new();
        for v in other {
            let _ = set.insert(v);
        }
        set
    }

    /// 判断是否包含指定值。
    pub fn contains(&self, value: V) -> bool {
        self.collex.contains(&value)
    }

    /// 返回最小元素。
    ///
    /// 集合为空时返回 `None`。
    pub fn first(&self) -> Option<V> {
        self.collex.first().map(|e| e.0)
    }

    /// 返回最大元素。
    ///
    /// 集合为空时返回 `None`。
    pub fn last(&self) -> Option<V> {
        self.collex.last().map(|e| e.0)
    }

    /// 插入一个值。
    ///
    /// ## 返回值
    /// - `Ok(())` — 插入成功
    /// - `Err(value)` — 值已存在或为负数
    pub fn insert(&mut self, value: V) -> Result<(), V> {
        self.collex.insert(SetElem(value)).map_err(|e| e.0)
    }

    /// 删除指定值。
    ///
    /// ## 返回值
    /// - `Ok(value)` — 删除成功
    /// - `Err(())` — 值不存在
    #[allow(clippy::result_unit_err)]
    pub fn remove(&mut self, value: V) -> Result<V, ()> {
        self.collex.remove(&value).map(|e| e.0)
    }

    /// 查找第一个 `> target` 的元素。
    pub fn find_gt(&self, target: V) -> Option<V> {
        self.collex.find_gt(&target).map(|e| e.0)
    }

    /// 查找第一个 `>= target` 的元素。
    pub fn find_ge(&self, target: V) -> Option<V> {
        self.collex.find_ge(&target).map(|e| e.0)
    }

    /// 查找最后一个 `< target` 的元素。
    pub fn find_lt(&self, target: V) -> Option<V> {
        self.collex.find_lt(&target).map(|e| e.0)
    }

    /// 查找最后一个 `<= target` 的元素。
    pub fn find_le(&self, target: V) -> Option<V> {
        self.collex.find_le(&target).map(|e| e.0)
    }

    /// 查找距离 `target` 最近的值。
    ///
    /// 优先返回精确匹配；等距时取较小值。
    pub fn find_closest(&self, target: V) -> Option<V> {
        let le = self.find_le(target);
        let ge = self.find_ge(target);
        match (le, ge) {
            (Some(l), Some(g)) => {
                if target - l <= g - target {
                    Some(l)
                } else {
                    Some(g)
                }
            }
            (Some(l), None) => Some(l),
            (None, Some(g)) => Some(g),
            (None, None) => None,
        }
    }

    /// 返回元素总数。
    pub fn len(&self) -> usize {
        self.collex.len()
    }

    /// 判断集合是否为空。
    pub fn is_empty(&self) -> bool {
        self.collex.is_empty()
    }

    /// 返回不可变迭代器，按值升序遍历。
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.collex.iter().map(|e| e.0)
    }

    /// 批量插入，自动忽略重复和负数。
    pub fn extend(&mut self, vec: Vec<V>) {
        for v in vec {
            let _ = self.insert(v);
        }
    }

    /// 批量插入并返回详细结果。
    ///
    /// 返回 `(成功插入的值列表, 被拒绝的值列表)`。
    pub fn try_extend(&mut self, vec: Vec<V>) -> (Vec<V>, Vec<V>) {
        let mut accepted = Vec::with_capacity(vec.len());
        let mut rejected = Vec::new();
        for v in vec {
            match self.insert(v) {
                Ok(()) => accepted.push(v),
                Err(e) => rejected.push(e),
            }
        }
        (accepted, rejected)
    }

    /// 修改指定值的元素，值改变后若无法插入新位置则排出元素。
    ///
    /// 参见 [`Collex::modify`](crate::Collex::modify)。
    pub fn modify<F, R>(&mut self, value: V, op: F) -> Result<R, crate::collex::ModifyError<R, SetElem<V>>>
    where
        F: FnOnce(&mut V) -> R
    {
        self.collex.modify(&value, |elem| op(&mut elem.0))
    }

    /// 尝试修改指定值的元素，失败时自动回滚。
    ///
    /// 参见 [`Collex::try_modify`](crate::Collex::try_modify)。
    #[allow(clippy::result_unit_err)]
    pub fn try_modify<F, R>(&mut self, value: V, op: F) -> Result<R, ()>
    where
        F: FnOnce(&mut V) -> R
    {
        self.collex.try_modify(&value, |elem| op(&mut elem.0))
    }
}

// ===================== Default =====================

impl<V: FieldValue + ConstUnit> Default for FieldSet<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ===================== IntoIterator =====================

impl<V: FieldValue + ConstUnit> IntoIterator for FieldSet<V> {
    type Item = V;
    type IntoIter = std::iter::Map<
        crate::collex::iter::IntoIter<SetElem<V>, V>,
        fn(SetElem<V>) -> V,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.collex.into_iter().map(|e| e.0)
    }
}

// ===================== 测试 =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let set = FieldSet::<u32>::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        // with_elements
        let set = FieldSet::with_elements(vec![5, 15, 25]);
        assert_eq!(set.len(), 3);
        assert_eq!(set.first(), Some(5));
        assert_eq!(set.last(), Some(25));
    }

    #[test]
    fn test_insert_contains() {
        let mut set = FieldSet::<u32>::new();

        assert!(set.insert(5).is_ok());
        assert!(set.insert(15).is_ok());

        assert!(set.contains(5));
        assert!(set.contains(15));
        assert!(!set.contains(25));

        assert_eq!(set.first(), Some(5));
        assert_eq!(set.last(), Some(15));
        assert!(!set.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut set = FieldSet::<u32>::new();
        set.insert(5).unwrap();

        let removed = set.remove(5).unwrap();
        assert_eq!(removed, 5);
        assert!(!set.contains(5));
        assert!(set.is_empty());

        // 删除不存在的值
        assert!(set.remove(10).is_err());
    }

    #[test]
    fn test_extend_try_extend() {
        let mut set = FieldSet::<u32>::new();

        // extend
        set.extend(vec![5, 15, 25]);
        assert!(set.contains(5));
        assert!(set.contains(15));
        assert_eq!(set.len(), 3);

        // try_extend
        let (accepted, rejected) = set.try_extend(vec![25, 35]);
        assert_eq!(accepted, vec![35]); // 25 重复被拒绝
        assert_eq!(rejected, vec![25]);
        assert!(set.contains(35));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_find_methods() {
        let mut set = FieldSet::<u32>::new();
        for v in [5u32, 15, 25] {
            set.insert(v).unwrap();
        }

        assert_eq!(set.find_gt(10), Some(15));
        assert_eq!(set.find_ge(15), Some(15));
        assert_eq!(set.find_lt(20), Some(15));
        assert_eq!(set.find_le(25), Some(25));
        assert_eq!(set.find_gt(30), None);
    }

    #[test]
    fn test_find_closest() {
        let mut set = FieldSet::<u32>::new();
        for v in [5u32, 15, 25] {
            set.insert(v).unwrap();
        }

        // 精确匹配
        assert_eq!(set.find_closest(5), Some(5));
        assert_eq!(set.find_closest(25), Some(25));

        // 等距取小
        assert_eq!(set.find_closest(10), Some(5));
        assert_eq!(set.find_closest(20), Some(15));

        // 边界
        assert_eq!(set.find_closest(0), Some(5));
        assert_eq!(set.find_closest(100), Some(25));
    }

    #[test]
    fn test_iter() {
        let mut set = FieldSet::<u32>::new();
        set.extend(vec![5, 15, 25]);

        let collected: Vec<_> = set.iter().collect();
        assert_eq!(collected, vec![5, 15, 25]);

        // IntoIterator（消耗所有权）
        let collected: Vec<_> = set.into_iter().collect();
        assert_eq!(collected, vec![5, 15, 25]);
    }

    #[test]
    fn test_duplicate() {
        let mut set = FieldSet::<u32>::new();
        assert!(set.insert(5).is_ok());
        assert!(set.insert(5).is_err()); // 重复
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_negative_ignored() {
        // 负数应被拒绝
        let mut set = FieldSet::<i32>::new();
        assert!(set.insert(-1).is_err());
        assert!(set.insert(0).is_ok());
        assert!(!set.is_empty());
        assert_eq!(set.first(), Some(0));
    }

    #[test]
    fn test_modify() {
        let mut set = FieldSet::<u32>::new();
        set.extend(vec![5, 25, 35, 45]);

        assert_eq!(set.first(), Some(5));

        // 值未修改
        set.modify(5, |_| ()).unwrap();
        assert_eq!(set.first(), Some(5));

        // 修改值但仍在此位置（不改块）
        set.modify(5, |v| *v = 1).unwrap();
        assert_eq!(set.first(), Some(1));

        // 修改值改变位置（向后移）
        set.modify(1, |v| *v = 15).unwrap();
        assert_eq!(set.first(), Some(15));

        // 修改值改变位置（向后移）
        set.modify(15, |v| *v = 26).unwrap();
        assert_eq!(set.first(), Some(25));

        // 改到最前面
        set.modify(45, |v| *v = 0).unwrap();
        assert_eq!(set.first(), Some(0));

        // 新值冲突（重复），应弹出元素
        let ans = set.modify(25, |v| *v = 26);
        assert!(ans.is_err());
        // 25 已被排出，26 还在
        assert!(!set.contains(25));
        assert!(set.contains(26));

        // modify 未找到
        let ans = set.modify::<_, ()>(100, |_| {});
        assert!(ans.is_err());
    }

    #[test]
    fn test_try_modify() {
        let mut set = FieldSet::<u32>::new();
        set.extend(vec![5, 25, 35]);

        // 值未修改
        set.try_modify(5, |_| ()).unwrap();
        assert_eq!(set.first(), Some(5));

        // 修改值改变位置
        set.try_modify(5, |v| *v = 15).unwrap();
        assert_eq!(set.first(), Some(15));
        assert!(!set.contains(5));

        // 新值冲突（重复），应回滚
        let ans = set.try_modify(25, |v| *v = 35);
        assert!(ans.is_err());
        // 回滚后 25 仍在原位
        assert!(set.contains(25));
        assert!(set.contains(35));

        // 尝试改到更大的值
        let _ans = set.try_modify::<_, ()>(25, |v| *v = 999);
        // 999 不在集合内且 >=0，insert 成功
        assert!(set.contains(999));
        assert!(!set.contains(25));
    }

    #[test]
    fn test_try_modify_rollback() {
        let mut set = FieldSet::<i32>::new();
        set.extend(vec![5, 15, 25]);

        // try_modify 改负数应回滚
        let ans = set.try_modify::<_, ()>(5, |v| *v = -1);
        assert!(ans.is_err());
        assert!(set.contains(5));
        assert!(!set.contains(-1));
        assert_eq!(set.len(), 3);
    }
}
