use flag_cell::{FlagCell, FlagRef};

pub mod map;
pub mod set;
pub mod collex;

/// 一个块。详见 具体容器类型 。
///
/// Thing：本块有元素 <br>
/// Prev ：本块无元素，有前一个非空块，存其引用 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者引用 <br>
/// Next ：本块无元素，有后一个非空块，存其引用 <br>
/// Void ：容器完全无任何元素 <br>
///
/// Hint：外部 **never** 提取内部类型。随意获取引用可能导致触发panic，或导致Map无法正常工作。
#[derive(Debug)]
enum RawField<K, V>
where K:Copy
{
    Thing((usize,K,FlagCell<V>)),
    Prev ((usize,FlagRef<V>)),
    Among((usize,FlagRef<V>), (usize, FlagRef<V>)),
    Next ((usize,FlagRef<V>)),
    Void,
}

impl<K, V> RawField<K, V>
where K:Copy
{
    
    pub fn as_thing(&self) -> &(usize,K,FlagCell<V>) {
        match self {
            Self::Thing(t) => t,
            _ => panic!("Called `RawField::into_thing()` on a not `Thing` value`"),
        }
    }
    
    pub fn into_thing(self) -> FlagCell<V> {
        match self {
            Self::Thing((..,c)) => c,
            _ => panic!("Called `RawField::into_thing()` on a not `Thing` value`"),
        }
    }
    
    pub fn prev(tuple: (usize,FlagRef<V>) ) -> Self{
        Self::Prev(tuple)
    }
    
    pub fn next(tuple: (usize,FlagRef<V>) ) -> Self{
        Self::Next(tuple)
    }
    
    pub fn among(prev: (usize,FlagRef<V>), next:  (usize,FlagRef<V>)) -> Self{
        Self::Among(prev,next)
    }
    
    pub fn void() -> Self {
        Self::Void
    }
    
    pub fn partial_clone(&self) -> Option<RawField<K, V>> {
        match self {
            RawField::Thing(..)
            => None,
            RawField::Prev(prev)
            => Some(Self::prev(prev.clone())),
            RawField::Among(prev, next)
            => Some(Self::among(prev.clone(), next.clone())),
            RawField::Next(next)
            => Some(Self::next(next.clone())),
            RawField::Void
            => Some(RawField::Void),
        }
    }
    
    pub fn prev_from(tuple: &(usize,K,FlagCell<V>)) -> Self{
        Self::Prev((tuple.0,tuple.2.flag_borrow()))
    }
    
    pub fn next_from(tuple: &(usize,K,FlagCell<V>)) -> Self{
        Self::Next((tuple.0,tuple.2.flag_borrow()))
    }
    
    
    pub fn borrow_prev_or_clone(&self) -> RawField<K, V> {
        match self {
            RawField::Thing(thing)
            => Self::prev_from(thing),
            RawField::Prev(prev)
            => Self::prev(prev.clone()),
            RawField::Among(prev, next)
            => Self::among(prev.clone(), next.clone()),
            RawField::Next(next)
            => Self::next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
    
    pub fn borrow_next_or_clone(&self) -> RawField<K, V> {
        match self {
            RawField::Thing(thing)
            => Self::next_from(thing),
            RawField::Prev(prev)
            => Self::prev(prev.clone()),
            RawField::Among(prev, next)
            => Self::among(prev.clone(), next.clone()),
            RawField::Next(next)
            => Self::next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
    
    /// 解包得到内部值
    ///
    /// 若非Thing，或正在被借用，返回Err返还self
    #[allow(dead_code)]
    pub fn try_unwrap(self) -> Result<V,Self> {
        if let RawField::Thing(t) = self {
            t.2.try_unwrap().map_err(|cell| RawField::Thing((t.0,t.1,cell)))
        } else {
            Err(self)
        }
    }
    
    
    /// 解包得到内部值
    ///
    /// # Panics
    /// 若非Thing，或正在被借用，panic
    pub fn unwrap(self) -> V {
        if let RawField::Thing(t) = self {
            t.2.unwrap()
        } else {
            panic!("called `RawField::unwrap()` on a not `Thing` value")
        }
    }
}

impl<K: Copy,T> Clone for RawField<K,T> {
    /// 克隆内部引用
    ///
    /// # Panics
    /// 若为 `Thing`，panic
    fn clone(&self) -> Self {
        match self {
            RawField::Thing(_)
            => panic!("Called `RawField::clone()` on a `Thing` value"),
            RawField::Prev(prev)
            => Self::prev(prev.clone()),
            RawField::Among(prev, next)
            => Self::among(prev.clone(), next.clone()),
            RawField::Next(next)
            => Self::next(next.clone()),
            RawField::Void
            => RawField::Void,
        }
    }
}