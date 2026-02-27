use crate::collex::CollexField;
use crate::{Collexetable, Field, FieldCollex, FieldValue, RawField};

// ========== 无 dyn 动态分发的不可变迭代器 ==========
/// 递归迭代栈的元素类型（静态类型，无动态分发）
enum IterStackItem<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    // 外层 items 的迭代器
    Outer(std::slice::Iter<'a, CollexField<E, V>>),
    // 子 Collex 的迭代器
    Inner(Iter<'a, E, V>),
}

/// `FieldCollex` 的不可变迭代器（纯静态类型，无 dyn）
pub struct Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    // 静态类型的迭代栈（替代 dyn 动态分发）
    stack: Vec<IterStackItem<'a, E, V>>,
}

impl<'a, E, V> Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 构造引用迭代器（内部调用）
    pub(crate) fn new(collex: &'a FieldCollex<E, V>) -> Self {
        Self {
            stack: vec![IterStackItem::Outer(collex.items.iter())],
        }
    }
}


impl<E, V> FieldCollex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 获取不可变迭代器
    pub fn iter(&self) -> Iter<'_, E, V> {
        Iter {
            stack: vec![IterStackItem::Outer(self.items.iter())],
        }
    }
}

// ========== 纯静态类型的 Iterator trait 实现 ==========
impl<'a, E, V> Iterator for Iter<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = &'a E;
    
    fn next(&mut self) -> Option<Self::Item> {
        // 从栈顶取出迭代器处理（栈式递归）
        while let Some(mut iter_item) = self.stack.pop() {
            match &mut iter_item {
                // 处理外层 items 迭代器
                IterStackItem::Outer(outer_iter) => {
                    while let Some(field) = outer_iter.next() {
                        match field {
                            RawField::Thing((_, field)) => match field {
                                // 单个元素，直接返回
                                Field::Elem(e) => {
                                    // 把当前外层迭代器放回栈（后续继续处理）
                                    self.stack.push(iter_item);
                                    return Some(e);
                                }
                                // 子 Collex，创建子迭代器并压入栈（优先处理子迭代器）
                                Field::Collex(collex) => {
                                    // 先把当前外层迭代器放回栈
                                    self.stack.push(iter_item);
                                    // 把子 Collex 迭代器压入栈（栈顶优先处理）
                                    self.stack.push(IterStackItem::Inner(collex.iter()));
                                    // 重新进入循环，处理子迭代器
                                    break;
                                }
                            },
                            // 空块，跳过
                            _ => continue,
                        }
                    }
                    // 外层迭代器处理完，无需放回栈
                }
                // 处理子 Collex 迭代器
                IterStackItem::Inner(inner_iter) => {
                    if let Some(item) = inner_iter.next() {
                        // 子迭代器未处理完，放回栈
                        self.stack.push(iter_item);
                        return Some(item);
                    }
                    // 子迭代器处理完，无需放回栈
                }
            }
        }
        
        // 所有迭代器处理完毕
        None
    }
}

#[derive(Debug)]
pub enum IntoIterStackItem<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    // 持有 vec::IntoIter 所有权（消耗型，不可克隆）
    Outer(std::vec::IntoIter<CollexField<E, V>>),
    // 持有子 Collex 的 IntoIter 所有权
    Inner(IntoIter<E, V>),
}

#[derive(Debug)]
pub struct IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    stack: Vec<IntoIterStackItem<E, V>>,
}

impl<E, V> IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    pub(crate) fn new(collex: FieldCollex<E, V>) -> Self {
        // 转移 items 所有权到 vec::IntoIter（消耗原 FieldCollex 的 items）
        let outer_iter = collex.items.into_iter();
        Self {
            stack: vec![IntoIterStackItem::Outer(outer_iter)],
        }
    }
}

// 核心修复：无克隆、真正消耗所有权的 Iterator 实现
impl<E, V> Iterator for IntoIter<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = E;
    
    fn next(&mut self) -> Option<Self::Item> {
        // 循环处理栈顶迭代器（所有权转移）
        while let Some(mut iter_item) = self.stack.pop() {
            match &mut iter_item {
                IntoIterStackItem::Outer(outer_iter) => {
                    // 遍历外层迭代器（消耗式）
                    while let Some(field) = outer_iter.next() {
                        match field {
                            RawField::Thing((_, field_in)) => match field_in {
                                // 匹配到元素：直接返回所有权，同时把剩余迭代器放回栈
                                Field::Elem(e) => {
                                    // 把「剩余未处理的外层迭代器」放回栈（无克隆，转移剩余所有权）
                                    self.stack.push(IntoIterStackItem::Outer(std::mem::take(outer_iter)));
                                    return Some(e);
                                }
                                // 匹配到子 Collex：转移子 Collex 所有权，创建子迭代器压栈
                                Field::Collex(collex) => {
                                    // 1. 把当前剩余的外层迭代器放回栈
                                    self.stack.push(IntoIterStackItem::Outer(std::mem::take(outer_iter)));
                                    // 2. 把子 Collex 迭代器压入栈（优先处理子迭代器）
                                    self.stack.push(IntoIterStackItem::Inner(IntoIter::new(collex)));
                                    // 跳出当前循环，处理子迭代器
                                    break;
                                }
                            },
                            // 空块：跳过，继续遍历外层迭代器
                            RawField::Prev(_) | RawField::Among(_, _) | RawField::Next(_) | RawField::Void => continue,
                        }
                    }
                    // 外层迭代器已耗尽，无需放回栈
                }
                IntoIterStackItem::Inner(inner_iter) => {
                    // 处理子迭代器（消耗式）
                    if let Some(item) = inner_iter.next() {
                        // 子迭代器未耗尽，放回栈继续处理
                        self.stack.push(iter_item);
                        return Some(item);
                    }
                    // 子迭代器已耗尽，无需放回栈
                }
            }
        }
        // 所有迭代器处理完毕
        None
    }
}

/// FieldCollex 所有权转移的 IntoIterator 实现（核心：消耗 self）
impl<E, V> IntoIterator for FieldCollex<E, V>
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

impl<'a, E, V> IntoIterator for &'a FieldCollex<E, V>
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

#[cfg(test)]
mod tests {
    use super::*;
    use span_core::Span;
    
    // ===================== 测试用元素类型（实现Collexetable<u32>） =====================
    #[derive(Debug, PartialEq, Eq, Ord, PartialOrd)]
    #[derive(Clone)]
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
        let span = Span::new_finite(0u32, 100u32);
        let unit = 20u32;
        let elems = vec![TestElem(5), TestElem(15), TestElem(25), TestElem(55)];
        let collex = FieldCollex::<TestElem, u32>::with_elements(span, unit, elems).unwrap();
        
        // 遍历验证顺序和内容
        let collected: Vec<_> = collex.iter().cloned().collect();
        assert_eq!(collected, vec![TestElem(5), TestElem(15), TestElem(25), TestElem(55)]);
        
        // for 循环遍历（IntoIterator）
        let mut values = Vec::new();
        for elem in &collex {
            values.push(elem.0);
        }
        assert_eq!(values, vec![5, 15, 25, 55]);
        
        // 空集合迭代器
        let empty_collex = FieldCollex::<TestElem, u32>::new(Span::new_finite(0u32, 100u32), 10u32).unwrap();
        assert!(empty_collex.iter().next().is_none());
    }
}