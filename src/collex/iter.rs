use crate::{Collexetable, Field, FieldCollex, FieldValue, RawField};
use crate::collex::CollexField;

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

impl<E, V> FieldCollex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 获取不可变迭代器（纯静态类型，无动态分发）
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

// ========== 安全的 IntoIterator 实现 ==========
impl<'a, E, V> IntoIterator for &'a FieldCollex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    type Item = &'a E;
    type IntoIter = Iter<'a, E, V>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
    #[test]
    fn test_static_iter() {
        // 测试纯静态迭代器（无 dyn）的遍历逻辑
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