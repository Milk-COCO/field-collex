pub mod iter;
pub mod se;

use std::{mem, ptr};
use crate::{Collexetable, ConstUnit, FieldValue};

// ===================== Slot =====================

/// 一个槽位，可存储 0、1 或多个元素。
///
/// 槽位之间通过 `prev` / `next` 形成非空槽链表，用于 O(1) 跳转。
/// 槽内如果为 `Many`，内部 Vec 始终按值升序排列。
#[derive(Debug, Clone)]
pub struct Slot<T> {
    /// 前一个有数据的槽位索引
    pub prev: Option<usize>,
    /// 后一个有数据的槽位索引
    pub next: Option<usize>,
    /// 当前槽位的值内容
    pub values: ValueCount<T>
}

impl<T> Slot<T> {
    /// 获取槽内最小元素的引用（空槽返回 None）
    pub fn first(&self) -> Option<&T> {
        match &self.values {
            ValueCount::Nope => {
                None
            }
            ValueCount::One(v) => {
                Some(v)
            }
            ValueCount::Many(vec) => {
                Some(vec.first()?)
            }
        }
    }

    /// 获取槽内最大元素的引用（空槽返回 None）
    pub fn last(&self) -> Option<&T> {
        match &self.values {
            ValueCount::Nope => {
                None
            }
            ValueCount::One(v) => {
                Some(v)
            }
            ValueCount::Many(vec) => {
                Some(vec.last()?)
            }
        }
    }
}

// ===================== ValueCount =====================

/// 槽内元素计数变体。
///
/// 三态：
/// - `Nope` — 空槽
/// - `One(T)` — 恰好一个元素
/// - `Many(Vec<T>)` — 多个元素（始终保持升序）
#[derive(Debug, Clone)]
pub enum ValueCount<T> {
    /// 空槽位
    Nope,
    /// 单元素
    One(T),
    /// 多元素（有序 Vec）
    Many(Vec<T>)
}

// ===================== Collex =====================

/// 基于分块槽位的有序集合。
///
/// 以 `unit = V::UNIT` 为步长将非负数值空间划分为等距槽位，
/// 每个槽位能存 0、1 或多个元素。通过 `value / unit` 可 O(1) 定位目标槽位，
/// 配合槽内二分查找和槽间 prev/next 指针实现高效的范围查询。
///
/// ## 类型参数
/// - `E`: 元素类型，需实现 [`Collexetable<V>`]
/// - `V`: 数值类型，需实现 [`FieldValue`] + [`ConstUnit`]
///
/// ## 特性
/// - **自动排序**：元素按 `collexate()` 值全局升序
/// - **自动去重**：插入已存在的值时返回 `Err(elem)`
/// - **负数过滤**：`collexate()` 值为负数时拒绝插入
/// - **序列化**：支持 `serde::Serialize` / `Deserialize`，序列化为纯数组
/// - **迭代**：实现 `IntoIterator`，支持 `for` 循环
///
/// ## 示例
/// ```ignore
/// use field_collex::{Collex, Collexetable};
///
/// let mut c = Collex::<MyElem, u32>::new();
/// c.insert(MyElem(5)).unwrap();
/// c.insert(MyElem(15)).unwrap();
///
/// assert_eq!(c.len(), 2);
/// assert_eq!(c.first().unwrap(), &MyElem(5));
/// assert_eq!(c.find_ge(&10).unwrap(), &MyElem(15));
/// ```
pub struct Collex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 槽位宽度
    pub(crate) unit: V,
    /// 槽位数组
    pub(crate) items: Vec<Slot<E>>,
}

impl<E: std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug for Collex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collex")
            .field("unit", &self.unit)
            .field("items", &self.items)
            .finish()
    }
}

impl<E: Clone, V: Clone> Clone for Collex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    fn clone(&self) -> Self {
        Self {
            unit: self.unit,
            items: self.items.clone(),
        }
    }
}

impl<E, V> Default for Collex<E, V>
where
    E: Collexetable<V>,
    V: FieldValue + ConstUnit,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<E,V> Collex<E,V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    /// 构造一个空集合。
    ///
    /// `unit` 取 `V::UNIT`（整数类型默认为 1）。
    pub fn new() -> Collex<E,V>
    where V:ConstUnit
    {
        Self {
            unit: V::UNIT,
            items: Vec::new(),
        }
    }

    /// 返回当前槽位宽度。
    pub fn unit(&self) -> &V {
        &self.unit
    }

    /// 计算指定值对应的槽位索引。
    ///
    /// 此方法不做任何边界检查，仅机械地返回 `value / unit`。
    #[inline(always)]
    pub(crate) fn idx_of(&self, target: &V) -> usize {
        target.div(self.unit).into_usize()
    }

    /// 判断给定的槽位索引是否在 items 范围内。
    #[inline(always)]
    #[must_use]
    pub(crate) fn contains_idx(&self, idx: usize) -> bool {
        idx < self.items.len()
    }

    /// 判断集合是否包含指定值。
    #[must_use = "不用返回值你调用它干啥？"]
    pub fn contains(&self, value: &V) -> bool {
        let idx = self.idx_of(value);
        if self.contains_idx(idx) {
            let slot = &self.items[idx];
            match &slot.values {
                ValueCount::Nope => {
                    false
                }
                ValueCount::One(v) => {
                    value.eq(v.collexate_ref())
                }
                ValueCount::Many(vec) => {
                    vec.binary_search_by(|e| e.collexate_ref().cmp(value)).is_ok()
                }
            }
        } else { false }
    }

    /// 返回最小元素的引用（集合为空时返回 `None`）。
    pub fn first(&self) -> Option<&E> {
        let idx = self.first_nonempty_index()?;
        self.items[idx].first()
    }

    /// 返回最大元素的引用（集合为空时返回 `None`）。
    pub fn last(&self) -> Option<&E> {
        let idx = self.last_nonempty_index()?;
        self.items[idx].last()
    }

    /// 获取第一个非空槽位的索引。
    ///
    /// 若 `items[0]` 为空，则通过其 `next` 指针链定位。
    pub(crate) fn first_nonempty_index(&self) -> Option<usize> {
        if self.items.is_empty() { return None; }

        let first = &self.items[0];
        match first.values {
            ValueCount::Nope => first.next,
            _ => Some(0),
        }
    }

    /// 获取最后一个非空槽位的索引。
    ///
    /// 若末位槽为空，则通过其 `prev` 指针链定位。
    pub(crate) fn last_nonempty_index(&self) -> Option<usize> {
        if self.items.is_empty() { return None; }

        let last_idx = self.items.len() - 1;
        let last = &self.items[last_idx];
        match last.values {
            ValueCount::Nope => last.prev,
            _ => Some(last_idx),
        }
    }

    /// 插入一个元素。
    ///
    /// ## 返回值
    /// - `Ok(())` — 插入成功
    /// - `Err(elem)` — 元素已存在或其 `collexate()` 值为负数
    ///
    /// 插入时自动维护槽内有序性和槽间 prev/next 指针链。
    pub fn insert(&mut self, elem: E) -> Result<(),E> {
        let val_ref = elem.collexate_ref();
        if val_ref.lt(&V::zero()) {
            return Err(elem)
        }
        let may_idx = self.idx_of(val_ref);
        // 目标索引越界 -> 根据当前最后一个非空块计算前导 -> reserve resize push
        // 目标索引不越界 -> 填充
        let old_len = self.items.len();

        if may_idx < old_len {
            // 不越（同时杜绝len<=0）
            // 判断是否存在
            if self.contains(val_ref) {
                return Err(elem)
            }
            let mut should_replace = false;
            let slot = &mut self.items[may_idx];
            match slot.values {
                ValueCount::Nope => {
                    slot.values = ValueCount::One(elem);
                    should_replace = true;
                }
                ValueCount::One(ref mut elem_ptr) => {
                    unsafe {
                        let old_elem = ptr::read(elem_ptr);
                        ptr::write(&mut slot.values, ValueCount::Many(
                            if old_elem.collexate_ref().gt(val_ref) {
                                vec![elem, old_elem]
                            } else {
                                vec![old_elem, elem]
                            }
                        ));
                    }
                }
                ValueCount::Many(ref mut vec) => {
                    if vec.is_empty() {
                        vec.push(elem);
                    } else {
                        match vec.binary_search_by(|e| e.collexate_ref().cmp(val_ref)) {
                            Ok(_) => unreachable!(),
                            Err(pos) => vec.insert(pos, elem),
                        }
                    }
                }
            }

            // 为了规避借用检查，单独写外面
            if should_replace {

                // 左右遍历，更新两侧的prev next

                // 向前遍历：把所有空槽位的 next 改为当前索引，直到碰到有效值(也改)
                let mut i = may_idx;
                while i > 0 {
                    i -= 1;
                    let this_slot = &mut self.items[i];

                    this_slot.next = Some(may_idx);

                    if let ValueCount::Nope = this_slot.values {

                    } else {
                        break;
                    }
                }

                // 向后遍历：把所有空槽位的 prev 改为当前索引，直到碰到有效值(也改)
                let max = self.items.len() - 1;
                let mut i = may_idx;
                while i < max {
                    i += 1;
                    let this_slot = &mut self.items[i];

                    this_slot.prev = Some(may_idx);

                    if let ValueCount::Nope = this_slot.values {

                    } else {
                        break;
                    }
                }
            }
        } else {
            // 越界处理：修改未越界部分、预分配并铺设指针链
            let last_valid_idx = self.last_nonempty_index();

            if let Some(last_valid_idx) = last_valid_idx {
                // 从最后一个有效向后遍历全部
                // 把 最后有效它自己 和 后面所有空槽位 的 next 改为当前索引
                let max = self.items.len() - 1;
                let mut i = last_valid_idx;
                self.items[i].next = Some(may_idx);
                while i < max {
                    i += 1;
                    let this_slot = &mut self.items[i];
                    this_slot.next = Some(may_idx); // 更新 next
                    #[cfg(debug_assertions)]
                    if let ValueCount::Nope = this_slot.values {} else {
                        // 不可能再有 有效的 了。
                        unreachable!()
                    }
                }
            }

            self.items.reserve(may_idx + 1 - old_len);

            self.items.resize_with(may_idx, || Slot {
                prev: last_valid_idx,
                // 指向即将 push 的目标
                next: Some(may_idx),
                values: ValueCount::Nope,
            });

            self.items.push(Slot {
                prev: last_valid_idx,
                next: None,
                values: ValueCount::One(elem),
            });
        }
        Ok(())
    }

    /// 删除指定值对应的元素。
    ///
    /// ## 返回值
    /// - `Ok(elem)` — 删除成功，返回被删除的元素
    /// - `Err(())` — 值不存在或为负数
    ///
    /// 删除后自动维护 prev/next 指针链，`Many` 降为 1 元素时退化为 `One`。
    #[allow(clippy::result_unit_err)]
    pub fn remove(&mut self, value: &V) -> Result<E, ()> {
        if value.lt(&V::zero()) {
            return Err(());
        }
        let may_idx = self.idx_of(value);
        if !self.contains_idx(may_idx) {
            return Err(());
        }

        // 检查是否存在
        match &self.items[may_idx].values {
            ValueCount::Nope => return Err(()),
            ValueCount::One(v) => if !v.collexate_ref().eq(value) { return Err(()); },
            ValueCount::Many(vec) => {
                if vec.binary_search_by(|e| e.collexate_ref().cmp(value)).is_err() {
                    return Err(());
                }
            }
        }

        // 保存前后非空槽位指针
        let left = self.items[may_idx].prev;
        let right = self.items[may_idx].next;

        // 把 values 整个拿出来，slot.values 置为 Nope
        let slot = &mut self.items[may_idx];
        let old_values = mem::replace(&mut slot.values, ValueCount::Nope);

        match old_values {
            ValueCount::Many(mut vec) => {
                let pos = vec.binary_search_by(|e| e.collexate_ref().cmp(value)).unwrap();
                let removed = vec.remove(pos);
                // 如果只剩一个元素，转为 One（省掉 Vec 的堆分配）
                if vec.len() == 1 {
                    let remaining = vec.pop().unwrap();
                    slot.values = ValueCount::One(remaining);
                } else {
                    slot.values = ValueCount::Many(vec);
                }
                Ok(removed)
            }
            ValueCount::One(removed) => {
                // slot.values 已为 Nope，prev/next 仍然正确（就是 left/right）
                // 更新链表：从左侧所有空槽位中剔除本槽
                let mut i = may_idx;
                while i > 0 {
                    i -= 1;
                    let this_slot = &mut self.items[i];
                    this_slot.next = right;
                    if let ValueCount::Nope = this_slot.values {
                    } else {
                        break;
                    }
                }
                // 从右侧所有空槽位中剔除本槽
                let max = self.items.len() - 1;
                let mut i = may_idx;
                while i < max {
                    i += 1;
                    let this_slot = &mut self.items[i];
                    this_slot.prev = left;
                    if let ValueCount::Nope = this_slot.values {
                    } else {
                        break;
                    }
                }
                Ok(removed)
            }
            ValueCount::Nope => unreachable!(),
        }
    }

    /// 查找第一个 `collexate() >= value` 的元素。
    ///
    /// 返回 `None` 表示不存在大于等于目标值的元素（包括 value < 0 的情况）。
    pub fn find_ge(&self, value: &V) -> Option<&E> {
        if value.lt(&V::zero()) {
            return None
        }

        let idx = self.idx_of(value);

        if self.contains_idx(idx) {
            let slot = &self.items[idx];
            match &slot.values {
                ValueCount::Nope => {
                    // 为空跳至下一个有效块的最小值
                    // 我们的插入和remove逻辑确保next绝对正确
                    // 因为块间严格递增，next 的最小值必然 > value (因为 next 的索引 > idx，值更大)
                    Some(self.items[slot.next?].first().unwrap())
                }
                ValueCount::One(v) => {
                    if v.collexate_ref().ge(value) {
                        Some(v)
                    } else {
                        // 当前值 < value，跳 next
                        Some(self.items[slot.next?].first().unwrap())
                    }
                }
                ValueCount::Many(vec) => {
                    if vec.last()?.collexate_ref().ge(value) {
                        // 确保当前slot包含目标。二分查找第一个 >= value 的位置
                        match vec.binary_search_by(|e| e.collexate_ref().cmp(value)) {
                            Ok(pos) => {
                                // 找到了相等的，它就是第一个 >= 的
                                Some(&vec[pos])
                            }
                            Err(pos) => {
                                // 没找到相等的，pos 是第一个 > target 的位置
                                if pos < vec.len() {
                                    Some(&vec[pos])
                                } else {
                                    // SAFETY: 之前判断了。
                                    unreachable!()
                                }
                            }
                        }
                    } else {
                        // 外层剪枝：如果最大值 < value，直接跳 next
                        Some(self.items[slot.next?].first().unwrap())
                    }
                }
            }
        } else {
            // 目标值属越界值，再往后没值了
            None
        }
    }

    /// 查找第一个 `collexate() > value` 的元素。
    ///
    /// 返回 `None` 表示不存在严格大于目标值的元素（包括 value < 0 的情况）。
    pub fn find_gt(&self, value: &V) -> Option<&E> {
        if value.lt(&V::zero()) {
            return None;
        }

        let idx = self.idx_of(value);

        if self.contains_idx(idx) {
            let slot = &self.items[idx];
            match &slot.values {
                ValueCount::Nope => {
                    // 为空跳至下一个有效块的最小值
                    // 我们的插入和remove逻辑确保next绝对正确
                    // 因为块间严格递增，next 的最小值必然 > value (因为 next 的索引 > idx，值更大)
                    Some(self.items[slot.next?].first().unwrap())
                }
                ValueCount::One(v) => {
                    if v.collexate_ref().gt(value) {
                        Some(v)
                    } else {
                        // 当前值 <= value，跳 next
                        Some(self.items[slot.next?].first().unwrap())
                    }
                }
                ValueCount::Many(vec) => {
                    if vec.last().unwrap().collexate_ref().gt(value) {
                        // 确保当前slot包含目标。二分查找第一个 > value 的位置
                        match vec.binary_search_by(|e| e.collexate_ref().cmp(value)) {
                            Ok(pos) => {
                                // 找到了相等的，第一个 > value 的是 pos + 1
                                if pos + 1 < vec.len() {
                                    Some(&vec[pos + 1])
                                } else {
                                    // 但没有更大的了，跳 next
                                    Some(self.items[slot.next?].first().unwrap())
                                }
                            }
                            Err(pos) => {
                                // pos 是第一个 > value 的位置
                                Some(&vec[pos])
                            }
                        }
                    } else {
                        // 外层剪枝：如果最大值 <= value，直接跳 next
                        Some(self.items[slot.next?].first().unwrap())
                    }
                }
            }
        } else {
            None
        }
    }

    /// 查找最后一个 `collexate() <= value` 的元素。
    ///
    /// 若目标值超出最大槽位范围，返回集合中的最大元素。
    /// 返回 `None` 仅当 value < 0 时。
    pub fn find_le(&self, value: &V) -> Option<&E> {
        if value.lt(&V::zero()) {
            return None
        }

        let idx = self.idx_of(value);

        if self.contains_idx(idx) {
            let slot = &self.items[idx];
            match &slot.values {
                ValueCount::Nope => {
                    // 为空跳至上一个有效块的最大值
                    // 我们的插入和remove逻辑确保next绝对正确
                    // 因为块间严格递增，prev 的最小值必然 < value
                    Some(self.items[slot.prev?].last().unwrap())
                }
                ValueCount::One(v) => {
                    if v.collexate_ref().le(value) {
                        Some(v)
                    } else {
                        // 当前值 > value，跳 prev
                        Some(self.items[slot.prev?].last().unwrap())
                    }
                }
                ValueCount::Many(vec) => {
                    if vec.first()?.collexate_ref().le(value) {
                        // 确保当前slot包含目标。二分查找第一个 <= value 的位置
                        match vec.binary_search_by(|e| e.collexate_ref().cmp(value)) {
                            Ok(pos) => {
                                // 找到了相等的，它就是第一个 <= 的
                                Some(&vec[pos])
                            }
                            Err(pos) => {
                                // 没找到相等的，pos 是第一个 > target 的位置
                                if pos > 0 {
                                    // 那么 pos - 1 就是第一个 < target 的位置
                                    Some(&vec[pos-1])
                                } else {
                                    // SAFETY: 之前判断了。
                                    unreachable!()
                                }
                            }
                        }
                    } else {
                        // 外层剪枝：如果最小值 > value，直接跳 prev
                        Some(self.items[slot.prev?].last().unwrap())
                    }
                }
            }
        } else {
            // 目标值属越界值，往前全都是比它小的。
            self.last()
        }
    }

    /// 查找最后一个 `collexate() < value` 的元素。
    ///
    /// 若目标值超出最大槽位范围，返回集合中的最大元素。
    /// 返回 `None` 仅当 value < 0 时。
    pub fn find_lt(&self, value: &V) -> Option<&E> {
        if value.lt(&V::zero()) {
            return None
        }

        let idx = self.idx_of(value);

        if self.contains_idx(idx) {
            let slot = &self.items[idx];
            match &slot.values {
                ValueCount::Nope => {
                    // 为空跳至上一个有效块的最大值
                    // 我们的插入和remove逻辑确保next绝对正确
                    // 因为块间严格递增，prev 的最小值必然 < value
                    Some(self.items[slot.prev?].last().unwrap())
                }
                ValueCount::One(v) => {
                    if v.collexate_ref().lt(value) {
                        Some(v)
                    } else {
                        // 当前值 >= value，跳 prev
                        Some(self.items[slot.prev?].last().unwrap())
                    }
                }
                ValueCount::Many(vec) => {
                    if vec.first()?.collexate_ref().lt(value) {
                        // 确保当前slot包含目标。二分查找第一个 < value 的位置
                        match vec.binary_search_by(|e| e.collexate_ref().cmp(value)) {
                            Ok(pos) => {
                                // 找到了相等的，第一个 < value 的是 pos - 1
                                if pos > 0 {
                                    Some(&vec[pos - 1])
                                } else {
                                    // 但没有更大的了，跳 prev
                                    Some(self.items[slot.prev?].last().unwrap())
                                }
                            }
                            Err(pos) => {
                                // pos 是第一个 > target 的位置
                                if pos > 0 {
                                    // 那么 pos - 1 就是第一个 < target 的位置
                                    Some(&vec[pos-1])
                                } else {
                                    // SAFETY: 之前判断了。
                                    unreachable!()
                                }
                            }
                        }
                    } else {
                        // 外层剪枝：如果最小值 >= value，直接跳 prev
                        Some(self.items[slot.prev?].last().unwrap())
                    }
                }
            }
        } else {
            // 目标值属越界值，往前全都是比它小的。
            self.last()
        }
    }

    /// 一次遍历同时查找 `<= value` 和 `> value` 的元素，
    /// 避免重复的 `idx_of` 和槽位访问。
    /// 返回 `(prev, next)` 其中 prev 是最后一个 `<= value` 的元素，
    /// next 是第一个 `> value` 的元素。
    pub fn find_le_and_gt(&self, value: &V) -> (Option<&E>, Option<&E>) {
        if value.lt(&V::zero()) {
            return (None, self.first());
        }

        let idx = self.idx_of(value);

        if !self.contains_idx(idx) {
            return (self.last(), None);
        }

        let slot = &self.items[idx];
        match &slot.values {
            ValueCount::Nope => {
                // 用 match 替代 and_then 闭包，避免每帧上千次闭包分配
                let prev = match slot.prev {
                    Some(i) => self.items[i].last(),
                    None => None,
                };
                let next = match slot.next {
                    Some(i) => self.items[i].first(),
                    None => None,
                };
                (prev, next)
            }
            ValueCount::One(v) => {
                if v.collexate_ref().le(value) {
                    let next = match slot.next {
                        Some(i) => self.items[i].first(),
                        None => None,
                    };
                    (Some(v), next)
                } else {
                    let prev = match slot.prev {
                        Some(i) => self.items[i].last(),
                        None => None,
                    };
                    (prev, Some(v))
                }
            }
            ValueCount::Many(vec) => {
                match vec.binary_search_by(|e| e.collexate_ref().cmp(value)) {
                    Ok(pos) => {
                        let next = if pos + 1 < vec.len() {
                            Some(&vec[pos + 1])
                        } else {
                            match slot.next {
                                Some(i) => self.items[i].first(),
                                None => None,
                            }
                        };
                        (Some(&vec[pos]), next)
                    }
                    Err(pos) => {
                        let prev = if pos > 0 {
                            Some(&vec[pos - 1])
                        } else {
                            match slot.prev {
                                Some(i) => self.items[i].last(),
                                None => None,
                            }
                        };
                        let next = if pos < vec.len() {
                            Some(&vec[pos])
                        } else {
                            match slot.next {
                                Some(i) => self.items[i].first(),
                                None => None,
                            }
                        };
                        (prev, next)
                    }
                }
            }
        }
    }

    /// 修改指定值的元素。
    ///
    /// 闭包可修改元素（包括其 `collexate()` 值）。若 collexate 值改变，
    /// 元素将被移动到新位置。若新位置插入失败（重复/负数），返回错误并排出元素。
    ///
    /// ## 返回值
    /// - `Ok(result)` — 修改成功，`result` 为闭包的返回值
    /// - `Err(ModifyError::NotFound)` — 未找到目标元素
    /// - `Err(ModifyError::InsertError(result, elem))` — 值改变后无法插入新位置
    pub fn modify<F, R>(&mut self, value: &V, op: F) -> Result<R, ModifyError<R, E>>
    where
        F: FnOnce(&mut E) -> R
    {
        let mut elem = self.remove(value).map_err(|_| ModifyError::NotFound)?;
        let old_val = *elem.collexate_ref();
        let result = op(&mut elem);

        if elem.collexate_ref().eq(&old_val) {
            // 值未变，插回原位（必然成功）
            self.insert(elem).ok();
            Ok(result)
        } else {
            // 值改变，尝试插入新位置
            match self.insert(elem) {
                Ok(()) => Ok(result),
                Err(e) => Err(ModifyError::InsertError(result, e)),
            }
        }
    }

    /// 尝试修改指定值的元素，插入新位置失败时自动回滚。
    ///
    /// 与 [`modify`](Self::modify) 不同，若 collexate 值改变但新位置插入失败，
    /// 会自动恢复旧值并插回原位，保证集合一致性。
    ///
    /// ## 返回值
    /// - `Ok(result)` — 修改成功
    /// - `Err(())` — 未找到目标元素，或新值插入失败（已回滚）
    #[allow(clippy::result_unit_err)]
    pub fn try_modify<F, R>(&mut self, value: &V, op: F) -> Result<R, ()>
    where
        F: FnOnce(&mut E) -> R
    {
        let mut elem = self.remove(value).map_err(|_| ())?;
        let old_val = *elem.collexate_ref();
        let result = op(&mut elem);

        if elem.collexate_ref().eq(&old_val) {
            // 值未变，插回原位
            self.insert(elem).ok();
            Ok(result)
        } else {
            // 值改变，尝试插入新位置
            match self.insert(elem) {
                Ok(()) => Ok(result),
                Err(mut rejected) => {
                    // 回滚：恢复旧值，插回原位
                    *rejected.collexate_mut() = old_val;
                    self.insert(rejected).ok();
                    Err(())
                }
            }
        }
    }

}

// ===================== ModifyError =====================

/// [`Collex::modify`] 可能产生的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModifyError<R, E> {
    /// 未找到目标元素
    NotFound,
    /// 值改变后插入新位置失败（重复或负数），携带闭包结果和元素
    InsertError(R, E),
}

// ===================== CollexCursor =====================

/// 游标遍历器，用于单调递增的 beat 时间线下高效查找。
///
/// 假设每次调用 `step()` 的 beat 参数**非递减**（时间只前进），
/// 则游标渐进前移，无需每帧 O(n) 的二分搜索和 idx_of。
///
/// 内部存储 `(槽位索引, 槽内子索引)` 追踪当前位置。
pub struct CollexCursor<'a, E, V>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    collex: &'a Collex<E, V>,
    /// 上次返回的 prev 位置（槽索引, Many槽内的子索引）
    prev_pos: Option<(usize, usize)>,
}

impl<'a, E: Collexetable<V>, V: FieldValue> CollexCursor<'a, E, V> {
    /// 创建新游标，初始位置在第一个元素之前。
    pub fn new(collex: &'a Collex<E, V>) -> Self {
        Self { collex, prev_pos: None }
    }

    /// 从已有的槽位索引恢复游标位置（用于跨帧保存）
    pub fn from_pos(collex: &'a Collex<E, V>, pos: Option<(usize, usize)>) -> Self {
        Self { collex, prev_pos: pos }
    }

    /// 导出当前游标位置，用于跨帧保存
    pub fn pos(&self) -> Option<(usize, usize)> {
        self.prev_pos
    }

    /// 推进游标到 beat，返回 `(prev, next)`：
    /// - `prev`：最后一个 `collexate() <= beat` 的元素
    /// - `next`：第一个 `collexate() > beat` 的元素
    ///
    /// beat 必须**非递减**（每帧的时间单调递增）。
    pub fn step(&mut self, beat: &V) -> (Option<&'a E>, Option<&'a E>) {
        let collex = self.collex; // 重借用，保证返回值绑定 'a 而非 &mut self
        if beat.lt(&V::zero()) {
            return (None, collex.first());
        }

        let (mut slot_idx, mut sub_idx) = match self.prev_pos {
            Some(pos) => (pos.0, pos.1 + 1),
            None => match collex.first_nonempty_index() {
                Some(i) => (i, 0),
                None => return (None, None),
            },
        };

        'outer: loop {
            if slot_idx >= collex.items.len() {
                return (collex.last(), None);
            }
            let slot = &collex.items[slot_idx];
            match &slot.values {
                ValueCount::Nope => {
                    if let Some(ni) = slot.next {
                        slot_idx = ni;
                        sub_idx = 0;
                        continue 'outer;
                    }
                    return (collex.last(), None);
                }
                ValueCount::One(v) => {
                    if v.collexate_ref().le(beat) {
                        self.prev_pos = Some((slot_idx, 0));
                        if let Some(ni) = slot.next {
                            slot_idx = ni;
                            sub_idx = 0;
                            continue 'outer;
                        }
                        return (Some(v), None);
                    } else {
                        let prev = elem_at(collex, self.prev_pos);
                        return (prev, Some(v));
                    }
                }
                ValueCount::Many(vec) => {
                    while sub_idx < vec.len() {
                        let e = &vec[sub_idx];
                        if e.collexate_ref().le(beat) {
                            self.prev_pos = Some((slot_idx, sub_idx));
                            sub_idx += 1;
                        } else {
                            let prev = elem_at(collex, self.prev_pos);
                            return (prev, Some(e));
                        }
                    }
                    if let Some(ni) = slot.next {
                        slot_idx = ni;
                        sub_idx = 0;
                        continue 'outer;
                    }
                    return (vec.last(), None);
                }
            }
        }
    }
}

fn elem_at<'a, E, V>(collex: &'a Collex<E, V>, pos: Option<(usize, usize)>) -> Option<&'a E>
where E: Collexetable<V>, V: FieldValue {
    let (slot_idx, sub_idx) = pos?;
    Some(match &collex.items[slot_idx].values {
        ValueCount::One(v) => v,
        ValueCount::Many(vec) => &vec[sub_idx],
        ValueCount::Nope => unreachable!(),
    })
}
