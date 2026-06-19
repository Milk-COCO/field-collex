use super::{Collex, Slot, ValueCount};
use crate::{Collexetable, FieldValue};
use std::marker::PhantomData;

// ===================== Iter =====================

/// [`Collex`] 的不可变引用迭代器。
///
/// 按槽位顺序线性扫描所有非空 `Slot` 中的元素，槽内 `Many` 块按升序逐个产出。
/// 时间复杂度：平摊 O(1) 每元素。
pub struct Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 所有槽位的切片
    items: &'a [Slot<E>],
    /// 当前槽位索引
    slot_idx: usize,
    /// 当前 Many 中的元素索引（仅当正在遍历 Many 时有效）
    many_idx: usize,
    /// 类型标记
    _phantom: PhantomData<V>,
}

impl<'a, E, V> Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 从 Collex 创建迭代器（内部使用）
    pub(crate) fn new(collex: &'a Collex<E, V>) -> Self {
        Self {
            items: &collex.items,
            slot_idx: 0,
            many_idx: 0,
            _phantom: PhantomData,
        }
    }
}

impl<'a, E, V> Iterator for Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let slot = self.items.get(self.slot_idx)?;
            match &slot.values {
                ValueCount::Nope => {
                    self.slot_idx += 1;
                    // 继续下一个槽
                }
                ValueCount::One(v) => {
                    self.slot_idx += 1;
                    return Some(v);
                }
                ValueCount::Many(vec) => {
                    if self.many_idx < vec.len() {
                        let elem = &vec[self.many_idx];
                        self.many_idx += 1;
                        return Some(elem);
                    }
                    // Many 耗尽，移到下一个槽
                    self.slot_idx += 1;
                    self.many_idx = 0;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // 粗略估计：剩余槽位数 到 剩余槽位数 * 64（宽松上限）
        let remaining_slots = self.items.len().saturating_sub(self.slot_idx);
        (0, Some(remaining_slots * 64))
    }
}

// ===================== IntoIter =====================

/// [`Collex`] 的所有权转移迭代器。
///
/// 消耗 `Collex`，按升序依次产出所有元素。逻辑与 `Iter` 一致，零额外分配。
pub struct IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 外层槽位迭代器（消耗所有权）
    items: std::vec::IntoIter<Slot<E>>,
    /// 当前 Many 中的元素迭代器
    current: std::vec::IntoIter<E>,
    /// 类型标记
    _phantom: PhantomData<V>,
}

impl<E, V> IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 从 Collex 创建所有权迭代器（内部使用）
    pub(crate) fn new(collex: Collex<E, V>) -> Self {
        Self {
            items: collex.items.into_iter(),
            current: Vec::new().into_iter(),
            _phantom: PhantomData,
        }
    }
}

impl<E, V> Iterator for IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        // 先尝试从当前 Many 迭代器中取元素
        if let Some(elem) = self.current.next() {
            return Some(elem);
        }

        // 当前 Many 耗尽，取下一个 Slot
        loop {
            let slot = self.items.next()?;
            match slot.values {
                ValueCount::Nope => {
                    // 空槽，继续
                }
                ValueCount::One(e) => {
                    return Some(e);
                }
                ValueCount::Many(vec) => {
                    self.current = vec.into_iter();
                    // 尝试从新 Many 中取第一个元素
                    if let Some(elem) = self.current.next() {
                        return Some(elem);
                    }
                    // 空 vec，继续下一个槽
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_slots = self.items.len();
        let current_remaining = self.current.len();
        (current_remaining, Some(remaining_slots * 64 + current_remaining))
    }
}

// ===================== IntoIterator =====================

impl<'a, E, V> IntoIterator for &'a Collex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = &'a E;
    type IntoIter = Iter<'a, E, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E, V> IntoIterator for Collex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = E;
    type IntoIter = IntoIter<E, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

// ===================== 便捷方法 =====================

impl<E, V> Collex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 返回不可变迭代器，按 `collexate()` 值升序遍历所有元素。
    pub fn iter(&self) -> Iter<'_, E, V> {
        Iter::new(self)
    }

    /// 返回集合中元素的总数。
    ///
    /// 时间复杂度 O(槽位数)，遍历所有槽累加。
    pub fn len(&self) -> usize {
        self.items
            .iter()
            .map(|slot| match &slot.values {
                ValueCount::Nope => 0,
                ValueCount::One(_) => 1,
                ValueCount::Many(vec) => vec.len(),
            })
            .sum()
    }

    /// 判断集合是否为空。
    ///
    /// 时间复杂度 O(1)。
    pub fn is_empty(&self) -> bool {
        self.first_nonempty_index().is_none()
    }
}

// ===================== 测试 =====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Collexetable;

    // 测试用元素类型
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

    #[test]
    fn test_iter() {
        let mut collex = Collex::<TestElem, u32>::new();
        collex.insert(TestElem(5)).unwrap();
        collex.insert(TestElem(15)).unwrap();
        collex.insert(TestElem(25)).unwrap();
        collex.insert(TestElem(55)).unwrap();

        // 遍历验证顺序和内容
        let collected: Vec<_> = collex.iter().cloned().collect();
        assert_eq!(
            collected,
            vec![TestElem(5), TestElem(15), TestElem(25), TestElem(55)]
        );

        // for 循环遍历（IntoIterator）
        let mut values = Vec::new();
        for elem in &collex {
            values.push(elem.0);
        }
        assert_eq!(values, vec![5, 15, 25, 55]);

        // 空集合迭代器
        let empty_collex = Collex::<TestElem, u32>::new();
        assert!(empty_collex.iter().next().is_none());
    }

    #[test]
    fn test_into_iter() {
        let mut collex = Collex::<TestElem, u32>::new();
        collex.insert(TestElem(10)).unwrap();
        collex.insert(TestElem(20)).unwrap();
        collex.insert(TestElem(30)).unwrap();

        // 所有权转移迭代
        let collected: Vec<_> = collex.into_iter().collect();
        assert_eq!(
            collected,
            vec![TestElem(10), TestElem(20), TestElem(30)]
        );
    }

    #[test]
    fn test_len_is_empty() {
        let mut collex = Collex::<TestElem, u32>::new();
        assert!(collex.is_empty());
        assert_eq!(collex.len(), 0);

        collex.insert(TestElem(5)).unwrap();
        assert!(!collex.is_empty());
        assert_eq!(collex.len(), 1);

        collex.insert(TestElem(15)).unwrap();
        assert_eq!(collex.len(), 2);

        collex.remove(&5).unwrap();
        assert_eq!(collex.len(), 1);

        collex.remove(&15).unwrap();
        assert!(collex.is_empty());
        assert_eq!(collex.len(), 0);
    }

    #[test]
    fn test_iter_skip_empty_slots() {
        // 测试跳槽插入导致的空槽不会被迭代器遍历到
        let mut collex = Collex::<TestElem, u32>::new();
        // 插入跨度大的元素，中间会留空槽
        collex.insert(TestElem(100)).unwrap();
        collex.insert(TestElem(5)).unwrap();

        // 迭代器只应返回非空元素，按值升序（5, 100）
        let collected: Vec<_> = collex.iter().map(|e| e.0).collect();
        assert_eq!(collected, vec![5, 100]);
        assert_eq!(collex.len(), 2);
    }
}
