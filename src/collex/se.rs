use super::Collex;
use crate::{Collexetable, FieldValue, ConstUnit};
use serde::{Serialize, Serializer, Deserialize, Deserializer};

// ===================== Serialize =====================

/// 序列化 [`Collex`] 为纯数组。
///
/// 遍历所有元素（按升序），序列化为 `[elem1, elem2, ...]` 格式。
/// 内部的槽位结构（`Slot`/`ValueCount`）不对外暴露。
impl<E, V> Serialize for Collex<E, V>
where
    E: Collexetable<V> + Serialize,
    V: FieldValue,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 利用迭代器收集所有元素为有序 Vec（仅序列化元素本身）
        let elements: Vec<&E> = self.iter().collect();
        elements.serialize(serializer)
    }
}

// ===================== Deserialize =====================

/// 从纯数组反序列化为 [`Collex`]。
///
/// 从 JSON 数组 `[elem1, elem2, ...]` 反序列化，逐个 `insert` 到新集合中。
/// - 重复元素自动跳过
/// - 负值自动过滤
/// - 不要求输入有序
impl<'de, E, V> Deserialize<'de> for Collex<E, V>
where
    E: Collexetable<V> + Deserialize<'de>,
    V: FieldValue + ConstUnit + 'static,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let elements: Vec<E> = Vec::deserialize(deserializer)?;
        let mut collex = Collex::new();

        // 直接逐个 insert，错误忽略（重复/负数跳过）
        for elem in elements {
            // insert 内部会检查重复和负数，失败静默忽略
            let _ = collex.insert(elem);
        }

        Ok(collex)
    }
}

// ===================== 测试 =====================

#[cfg(test)]
mod serialize_tests {
    use super::*;
    use serde_json::{from_str, to_string};
    use crate::Collexetable;

    // 测试用元素类型（可序列化）
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
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

    /// Collex → 纯数组 JSON
    #[test]
    fn test_serialize_to_array() {
        let mut collex = Collex::<TestElem, u32>::new();
        collex.insert(TestElem(5)).unwrap();
        collex.insert(TestElem(15)).unwrap();
        collex.insert(TestElem(25)).unwrap();

        let json = to_string(&collex).unwrap();
        assert_eq!(json, "[5,15,25]");
    }

    /// 纯数组 → Collex
    #[test]
    fn test_deserialize_from_array() {
        let json = r#"[5, 15, 25]"#;
        let collex: Collex<TestElem, u32> = from_str(json).unwrap();

        assert_eq!(collex.len(), 3);
        assert!(collex.contains(&5));
        assert!(collex.contains(&15));
        assert!(collex.contains(&25));
        assert_eq!(collex.first(), Some(&TestElem(5)));
        assert_eq!(collex.last(), Some(&TestElem(25)));
    }

    /// 反序列化 → 修改 → 再序列化
    #[test]
    fn test_modify_after_deserialize() {
        let json = r#"[5, 15, 25]"#;
        let mut collex: Collex<TestElem, u32> = from_str(json).unwrap();

        // 删除 15
        let removed = collex.remove(&15).unwrap();
        assert_eq!(removed, TestElem(15));

        // 重新序列化
        let modified_json = to_string(&collex).unwrap();
        assert_eq!(modified_json, "[5,25]");
    }

    /// 空集合序列化/反序列化
    #[test]
    fn test_empty_collex_serde() {
        let collex = Collex::<TestElem, u32>::new();
        let json = to_string(&collex).unwrap();
        assert_eq!(json, "[]");

        let deserialized: Collex<TestElem, u32> = from_str("[]").unwrap();
        assert!(deserialized.is_empty());
    }

    /// 重复元素反序列化时自动去重
    #[test]
    fn test_duplicate_on_deserialize() {
        let json = r#"[5, 5, 15, 15, 25]"#;
        let collex: Collex<TestElem, u32> = from_str(json).unwrap();

        // 重复元素被跳过，只保留一份
        assert_eq!(collex.len(), 3);
        assert!(collex.contains(&5));
        assert!(collex.contains(&15));
        assert!(collex.contains(&25));
    }
}
