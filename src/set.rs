use span_core::Span;
use crate::*;
use crate::collex::*;

type FieldIn<V> = Field<V,FieldSet<V>>;
type SetField<V> = RawField<Field<V,FieldSet<V>>>;

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SetElem<V: FieldValue>(V);

impl<V: FieldValue> Collexetable<V> for SetElem<V>
{
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

fn elem_to_v_vec<V: FieldValue>(vec: Vec<SetElem<V>>) -> Vec<V> {
    let mut vec = std::mem::ManuallyDrop::new(vec);
    
    let ptr = vec.as_mut_ptr() as *mut V;
    let len = vec.len();
    let cap = vec.capacity();
    unsafe {
        Vec::from_raw_parts(ptr, len, cap)
    }
}

fn v_to_elem_vec<V: FieldValue>(vec: Vec<V>) -> Vec<SetElem<V>> {
    let mut vec = std::mem::ManuallyDrop::new(vec);
    
    let ptr = vec.as_mut_ptr() as *mut SetElem<V>;
    let len = vec.len();
    let cap = vec.capacity();
    unsafe {
        Vec::from_raw_parts(ptr, len, cap)
    }
}

type In<V> = FieldCollex<SetElem<V>,V>;

/// 每个块可以存多个内容（通过递归结构实现）
/// 非空块可为单个元素或一个FieldSet，以[`Field`]类型存储。
#[derive(Debug)]
pub struct FieldSet<V: FieldValue>(In<V>);

impl<V> FieldSet<V>
where
    V: FieldValue,
{
    pub fn new(span: Span<V>, unit: V) -> NewResult<Self,V>
    {
        Ok(Self(In::new(span, unit)?))
    }
    pub fn with_capacity(
        span: Span<V>,
        unit: V,
        capacity: usize,
    ) -> WithCapacityResult<Self,V>
    {
        Ok(Self(In::with_capacity(span, unit, capacity)?))
    }
    pub fn with_elements(
        span: Span<V>,
        unit: V,
        other: Vec<V>,
    ) -> WithElementsResult<Self,V>
    {
        let other = v_to_elem_vec(other); 
        Ok(Self(In::with_elements(span, unit, other)?))
    }
    pub fn span(&self) -> &Span<V>
    {
        self.0.span()
    }
    
    pub fn unit(&self) -> &V
    {
        self.0.unit()
    }
    
    pub fn size(&self) -> Option<usize>
    {
        self.0.size()
    }
    
    pub fn len(&self) -> usize
    {
        self.0.len()
    }
    pub fn capacity(&self) -> usize
    {
        self.0.capacity()
    }
    
    pub fn contains(&self, value: V) -> bool
    {
        self.0.contains_value(value)
    }
    
    pub fn first(&self) -> Option<V>
    {
        self.0.first().map(|r|r.0)
    }
    
    pub fn last(&self) -> Option<V>
    {
        self.0.last().map(|r|r.0)
    }
    pub fn extend(&mut self, vec: Vec<V>)
    {
        let vec = v_to_elem_vec(vec); 
        self.0.extend(vec)
    }
    
    pub fn try_extend(&mut self, vec: Vec<V>) -> TryExtendResult<V> 
    {
        let vec = v_to_elem_vec(vec); 
        let ans = self.0.try_extend(vec);
        TryExtendResult {
            out_of_span: elem_to_v_vec(ans.out_of_span),
            already_exist: elem_to_v_vec(ans.already_exist),
        }
    }
    pub fn insert(&mut self, value: V) -> InsertResult<V>
    {
        let ans = self.0.insert(SetElem(value));
        
        ans.map_err(|e| match e {
            InsertFieldCollexError::OutOfSpan(elem) => InsertFieldCollexError::OutOfSpan(elem.0),
            InsertFieldCollexError::AlreadyExist(elem) => InsertFieldCollexError::AlreadyExist(elem.0),
        })
    }
    pub fn remove(&mut self, target: V) -> RemoveResult<V>
    {
        let ans = self.0.remove(target);
        
        ans.map(|elem| elem.0)
    }
    pub fn find_gt(&self, target: V) -> Option<V>
    {
        let ans = self.0.find_gt(target);
        ans.map(|elem| elem.0)
    }
    
    pub fn find_ge(&self, target: V) -> Option<V>
    {
        let ans = self.0.find_ge(target);
        ans.map(|elem| elem.0)
    }
    
    pub fn find_lt(&self, target: V) -> Option<V>
    {
        let ans = self.0.find_lt(target);
        ans.map(|elem| elem.0)
    }
    
    pub fn find_le(&self, target: V) -> Option<V>
    {
        let ans = self.0.find_le(target);
        ans.map(|elem| elem.0)
    }
    
    pub fn find_closest(&self, target: V) -> Option<V> {
        let ans = self.0.find_closest(target);
        ans.map(|elem| elem.0)
    }
    
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use span_core::Span;
    
    // ===================== Pub 方法测试用例 =====================
    #[test]
    fn test_basic_construction() {
        // 测试：new / with_capacity / with_elements / span / unit / size / len / capacity / is_empty
        let finite_span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        
        // 1. new 方法（正常场景）
        let set = FieldSet::<u32>::new(finite_span.clone(), unit).unwrap();
        assert_eq!(set.span(), &finite_span);
        assert_eq!(*set.unit(), unit);
        assert_eq!(set.size(), Some(10)); // 100/10=10 块
        assert_eq!(set.len(), 0);
        assert_eq!(set.capacity(), 0);
        assert!(set.is_empty());
        
        // 2. new 方法（错误场景：unit=0）
        let err_unit_zero = FieldSet::<u32>::new(finite_span.clone(), 0u32).unwrap_err();
        assert!(matches!(err_unit_zero, NewFieldCollexError::NonPositiveUnit(_, 0)));
        
        // 3. new 方法（错误场景：空 span）
        let empty_span = Span::new_finite(5u32, 3u32); // start >= end 为空
        let err_empty_span = FieldSet::<u32>::new(empty_span, unit).unwrap_err();
        assert!(matches!(err_empty_span, NewFieldCollexError::EmptySpan(_, _)));
        
        // 4. with_capacity 方法（正常场景）
        let set_with_cap = FieldSet::<u32>::with_capacity(finite_span, unit, 5).unwrap();
        assert_eq!(set_with_cap.capacity(), 5);
        assert!(set_with_cap.is_empty());
        
        // 5. with_capacity 方法（错误场景：capacity 超限）
        let err_cap_out = FieldSet::<u32>::with_capacity(Span::new_finite(0u32, 100u32), 10u32, 11).unwrap_err();
        assert!(matches!(err_cap_out, WithCapacityFieldCollexError::OutOfSize(_, _)));
    }
    
    #[test]
    fn test_insert_contains() {
        // 测试：insert / contains / first / last
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut set = FieldSet::<u32>::new(span, unit).unwrap();
        
        // 插入元素（直接传入 u32，无需包装）
        let val1 = 5u32;
        let val2 = 15u32;
        assert!(set.insert(val1).is_ok());
        assert!(set.insert(val2).is_ok());
        
        // 验证包含性
        assert!(set.contains(val1));
        assert!(set.contains(val2));
        assert!(!set.contains(25u32));
        
        // 验证首尾元素
        assert_eq!(set.first(), Some(val1));
        assert_eq!(set.last(), Some(val2));
        
        // 验证非空
        assert!(!set.is_empty());
    }
    
    #[test]
    fn test_remove() {
        // 测试：remove
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut set = FieldSet::<u32>::new(span, unit).unwrap();
        
        // 插入后删除
        let val = 5u32;
        set.insert(val).unwrap();
        let removed = set.remove(val).unwrap();
        assert_eq!(removed, val);
        
        // 验证删除后不包含
        assert!(!set.contains(val));
        assert!(set.is_empty());
        
        // 错误场景：删除不存在的值
        let err_remove = set.remove(10u32).unwrap_err();
        assert!(matches!(err_remove, RemoveFieldCollexError::NotExist));
    }
    
    #[test]
    fn test_extend_try_extend() {
        // 测试：extend / try_extend
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut set = FieldSet::<u32>::new(span, unit).unwrap();
        
        // 1. extend：批量插入（直接传入 Vec<u32>）
        let vals = vec![5u32, 15u32, 25u32];
        set.extend(vals.clone());
        assert!(set.contains(5u32));
        assert!(set.contains(15u32));
        
        // 2. try_extend：批量插入并返回结果
        let vals2 = vec![25u32, 35u32, 105u32]; // 105 超出 span 范围
        let result = set.try_extend(vals2);
        // 验证：105 超出 span，25 已存在，35 插入成功
        assert!(!result.out_of_span.is_empty() && result.out_of_span[0] == 105);
        assert!(!result.already_exist.is_empty() && result.already_exist[0] == 25);
        assert!(set.contains(35u32));
    }
    
    #[test]
    fn test_find_methods() {
        // 测试：find_gt / find_ge / find_lt / find_le
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let mut set = FieldSet::<u32>::new(span, unit).unwrap();
        
        // 插入测试元素
        let vals = [5u32, 15u32, 25u32];
        for &v in &vals {
            set.insert(v).unwrap();
        }
        
        // 测试 find_gt（大于）
        let gt = set.find_gt(10u32).unwrap();
        assert_eq!(gt, 15u32);
        
        // 测试 find_ge（大于等于）
        let ge = set.find_ge(15u32).unwrap();
        assert_eq!(ge, 15u32);
        
        // 测试 find_lt（小于）
        let lt = set.find_lt(20u32).unwrap();
        assert_eq!(lt, 15u32);
        
        // 测试 find_le（小于等于）
        let le = set.find_le(25u32).unwrap();
        assert_eq!(le, 25u32);
        
        // 错误场景：找不到匹配值
        let err_find = set.find_gt(30u32);
        assert!(matches!(err_find, None));
    }
    
    #[test]
    fn test_with_elements() {
        // 测试：with_elements（批量构造）
        let span = Span::new_finite(0u32, 100u32);
        let unit = 10u32;
        let vals = vec![5u32, 15u32, 25u32, 105u32]; // 105 超出 span
        
        // 构造 FieldSet（直接传入 Vec<u32>）
        let set = FieldSet::<u32>::with_elements(span, unit, vals).unwrap();
        // 验证：105 被忽略，5/15/25 插入成功
        assert!(set.contains(5u32));
        assert!(set.contains(15u32));
        assert!(!set.contains(105u32));
        assert_eq!(set.first(), Some(5u32));
        assert_eq!(set.last(), Some(25u32));
    }
}