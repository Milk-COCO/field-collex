use crate::{Collexetable, FieldCollex, FieldValue};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use span_core::Span;

impl<'a, E, V> Serialize for FieldCollex<E, V>
where
    E: Collexetable<V> + Serialize,
    V: FieldValue,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 利用迭代器收集所有元素为有序 Vec（核心：仅序列化元素本身）
        let elements: Vec<&E> = self.iter().collect();
        elements.serialize(serializer)
    }
}

/// 用于反序列化的辅助结构体（包含 Span、unit 和元素列表）
#[derive(Deserialize)]
pub struct FieldCollexSerdeHelper<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    pub span: Span<V>,
    pub unit: V,
    pub elements: Vec<E>,
}

impl<'de, E, V> Deserialize<'de> for FieldCollex<E, V>
where
    E: Collexetable<V> + Deserialize<'de>,
    V: FieldValue + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // 1. 先反序列化出辅助结构体
        let helper = FieldCollexSerdeHelper::<E, V>::deserialize(deserializer)?;
        
        // 2. 还原 FieldCollex
        Self::with_elements(helper.span, helper.unit, helper.elements)
            .map_err(|e| serde::de::Error::custom(format!("反序列化失败: {}", e)))
    }
}