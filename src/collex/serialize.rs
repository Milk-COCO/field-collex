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

pub fn default_span<V: FieldValue>() -> Span<V> {
    Span::new_infinite(V::zero())
}

pub fn default_unit<V: FieldValue>() -> V {
    V::from_usize(1)
}

pub fn default_elements<E>() -> Vec<E> {
    vec![]
}

/// 用于反序列化的辅助结构体（包含 Span、unit 和元素列表）
#[derive(Deserialize, Debug)]
pub struct FieldCollexSerdeHelper<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    #[serde(default = "default_span")]
    pub span: Span<V>,
    #[serde(default = "default_unit")]
    pub unit: V,
    #[serde(default = "default_elements")]
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
        let wrapper = FieldCollexSerdeWrapper::<E, V>::deserialize(deserializer)?;
        let helper: FieldCollexSerdeHelper<E, V> = wrapper.into();
        // 2. 还原 FieldCollex
        Self::with_elements(helper.span, helper.unit, helper.elements)
            .map_err(|e| serde::de::Error::custom(format!("反序列化失败: {}", e)))
    }
}

/// 无标签枚举：自动匹配「纯数组」或「结构体」格式
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum FieldCollexSerdeWrapper<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 纯数组格式（[...]）→ 映射到 elements 字段，span/unit 用默认值
    Array(Vec<E>),
    /// 结构体格式（{elements:..., span:...}
    Struct(FieldCollexSerdeHelper<E, V>),
}

impl<E, V> From<FieldCollexSerdeWrapper<E, V>> for FieldCollexSerdeHelper<E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    fn from(wrapper: FieldCollexSerdeWrapper<E, V>) -> Self {
        match wrapper {
            // 纯数组 → elements=数组，span/unit 用默认值
            FieldCollexSerdeWrapper::Array(elements) => Self {
                elements,
                span: default_span(),
                unit: default_unit(),
            },
            // 结构体 → 直接使用原有值
            FieldCollexSerdeWrapper::Struct(helper) => helper,
        }
    }
}

#[cfg(test)]
mod serialize_tests {
    // 导入核心模块和依赖
    use super::*;
    use serde_json::{from_str, to_string};
    use span_core::Span;
    use crate::{Field, RawField};

        // ===================== 测试基础定义 =====================
        /// 测试用元素类型（实现Collexetable<u32>，支持序列化/反序列化）
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
        
        // ===================== 核心测试用例（适配 Range/RangeFrom） =====================
        /// 测试1：基础序列化（FieldCollex → 纯数组JSON）
        #[test]
        fn test_serialize_to_array() {
            // 1. 构造FieldCollex（使用原生Span<Range/RangeFrom>）
            let finite_span = Span::Finite(0u32..100u32); // Fin=Range
            let unit = 10u32;
            let elems = vec![TestElem(5), TestElem(15), TestElem(25)];
            let collex = FieldCollex::<TestElem, u32>::with_elements(finite_span, unit, elems).unwrap();
            
            // 2. 序列化（仅输出元素数组）
            let json = to_string(&collex).unwrap();
            
            println!("{}", json);
            // 3. 验证结果
            assert_eq!(json, "[5,15,25]");
        }
        
        /// 测试2：纯数组反序列化（[...] → FieldCollex，默认无限Span）
        #[test]
        fn test_deserialize_from_array() {
            // 1. 纯数组JSON输入
            let json = r#"[5, 15, 25]"#;
            
            // 2. 反序列化为FieldCollex
            let collex: FieldCollex<TestElem, u32> = from_str(json).unwrap();
            
            // 3. 验证默认属性（Infinite=RangeFrom<0> + unit=1）
            assert!(matches!(collex.span(), Span::Infinite(r) if r.start == 0)); // 匹配RangeFrom
            assert_eq!(*collex.unit(), 1u32);
            
            // 验证元素（修复len()语义误解）
            assert!(collex.contains(&TestElem(5)));
            assert!(collex.contains(&TestElem(15)));
            assert!(collex.contains(&TestElem(25)));
            assert_eq!(count_elements(&collex), 3);
            
            // 4. 验证功能
            assert_eq!(collex.find_closest(20u32), Some(&TestElem(15)));
        }
        
        /// 测试3：结构体格式反序列化（适配 Range/RangeFrom 序列化格式）
        #[test]
        fn test_deserialize_from_struct() {
            // 1. 【核心修复】匹配Span<Range>的原生序列化格式：
            //    Span::Finite(0..200) → {"Finite": {"start":0, "end":200}}
            let json = r#"{
            "span": {
                "Finite": {
                    "start": 0,
                    "end": 200
                }
            },
            "unit": 5,
            "elements": [10, 20, 30]
        }"#;
            
            // 2. 反序列化（完全适配原生Span格式）
            let collex: FieldCollex<TestElem, u32> = from_str(json).unwrap();
            
            // 3. 验证核心属性
            assert!(matches!(collex.span(), Span::Finite(r)if r.start==0 && r.end == 200)); // 匹配Range
            assert_eq!(*collex.unit(), 5u32);
            assert_eq!(count_elements(&collex), 3);
            
            // 4. 验证功能
            assert_eq!(collex.first(), Some(&TestElem(10)));
            assert_eq!(collex.last(), Some(&TestElem(30)));
            assert_eq!(collex.find_gt(25u32), Some(&TestElem(30)));
        }
        
        /// 测试4：嵌套FieldCollex的序列化/反序列化
        #[test]
        fn test_serialize_deserialize_nested_collex() {
            // 1. 构造嵌套Collex（原生Span）
            let outer_span = Span::Finite(0u32..200u32);
            let outer_unit = 100u32;
            
            // 内层Collex
            let inner_span = Span::Finite(0u32..100u32);
            let inner_unit = 10u32;
            let inner_elems = vec![TestElem(5), TestElem(15)];
            let _inner_collex = FieldCollex::<TestElem, u32>::with_elements(inner_span, inner_unit, inner_elems).unwrap();
            
            // 外层Collex
            let outer_elems = vec![TestElem(105), TestElem(115)];
            let mut outer_collex = FieldCollex::<TestElem, u32>::new(outer_span, outer_unit).unwrap();
            outer_collex.extend(outer_elems);
            
            // 2. 序列化
            let json = to_string(&outer_collex).unwrap();
            assert_eq!(json, "[105,115]");
            
            // 3. 反序列化
            let deserialized: FieldCollex<TestElem, u32> = from_str(&json).unwrap();
            
            // 4. 验证
            assert!(deserialized.contains(&TestElem(105)));
            assert!(!deserialized.contains(&TestElem(5)));
        }
        
        /// 测试5：反序列化错误场景（非法span/unit）
        #[test]
        fn test_deserialize_invalid_input() {
            // 场景1：unit=0（非法）
            let json_invalid_unit = r#"{
            "span": {
                "Finite": {
                    "start": 0,
                    "end": 100
                }
            },
            "unit": 0,
            "elements": [5, 15]
        }"#;
            let err = from_str::<FieldCollex<TestElem, u32>>(json_invalid_unit);
            assert!(err.is_err());
            
            // 宽松断言（匹配核心关键词）
            let err_str = err.unwrap_err().to_string();
            println!("unit=0 错误信息: {}", err_str);
            assert!(err_str.contains("unit") && (err_str.contains("0") || err_str.contains("非正")));
            
            // 场景2：空span（start >= end）
            let json_empty_span = r#"{
            "span": {
                "Finite": {
                    "start": 50,
                    "end": 30
                }
            },
            "unit": 10,
            "elements": [5, 15]
        }"#;
            let err2 = from_str::<FieldCollex<TestElem, u32>>(json_empty_span);
            assert!(err2.is_err());
            
            let err2_str = err2.unwrap_err().to_string();
            println!("空span 错误信息: {}", err2_str);
            assert!(err2_str.contains("span") && err2_str.contains("空"));
            
            // 场景3：纯数组反序列化后插入超大元素（无限span允许）
            let json_valid_array = r#"[5, 15, 25]"#;
            let mut collex = from_str::<FieldCollex<TestElem, u32>>(json_valid_array).unwrap();
            assert!(collex.insert(TestElem(1000)).is_ok());
            assert!(collex.contains(&TestElem(1000)));
        }
        
        /// 测试6：反序列化后修改元素
        #[test]
        fn test_modify_after_deserialize() {
            // 1. 反序列化
            let json = r#"[5, 15, 25]"#;
            let mut collex: FieldCollex<TestElem, u32> = from_str(json).unwrap();
            
            // 2. 修改元素
            let result = collex.modify(15u32, |e| e.0 = 18);
            assert!(result.is_ok());
            
            // 3. 验证修改
            assert_eq!(collex.get(18u32), Some(&TestElem(18)));
            assert!(!collex.contains(&TestElem(15)));
            
            // 4. 重新序列化
            let modified_json = to_string(&collex).unwrap();
            assert_eq!(modified_json, "[5,18,25]");
        }
        
        // ===================== 辅助方法 =====================
        /// 统计FieldCollex中所有层级的元素数量（修复len()语义误解）
        fn count_elements<E, V>(collex: &FieldCollex<E, V>) -> usize
        where
            E: Collexetable<V>,
            V: FieldValue,
        {
            let mut count = 0;
            for item in &collex.items {
                match item {
                    RawField::Thing((_, Field::Elem(_))) => count += 1,
                    RawField::Thing((_, Field::Collex(sub_collex))) => {
                        count += count_elements(sub_collex);
                    }
                    _ => continue,
                }
            }
            count
        }
    }