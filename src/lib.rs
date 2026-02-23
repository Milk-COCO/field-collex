use num_traits::{NumOps, Zero};

pub mod collex;
pub mod set;

pub use set::FieldSet;
pub use collex::FieldCollex;
pub use collex::Collexetable;

pub trait FieldValue: Ord + Copy + Into<usize> + NumOps + Zero {
    fn ceil(&self) -> Self;
}

pub(crate) trait FieldItem<V> {
    fn first(&self) -> &V;
    fn last(&self) -> &V;
}

/// 一个块。详见 具体容器类型 。
///
/// Thing：本块有元素，存本块索引+值 <br>
/// Prev ：本块无元素，有前一个非空块，存其索引 <br>
/// Among：本块无元素，有前与后一个非空块，存其二者索引 <br>
/// Next ：本块无元素，有后一个非空块，存其索引 <br>
/// Void ：容器完全无任何元素 <br>
///
#[derive(Debug)]
pub enum RawField<V, IDX = usize> {
    Thing((IDX, V)),
    Prev (IDX),
    Among(IDX, IDX),
    Next (IDX),
    Void,
}

impl<V> RawField<V> {
    pub fn as_thing(&self) -> (usize, &V) {
        match self {
            Self::Thing(t) => (t.0,&t.1),
            _ => panic!("Called `RawField::as_thing()` on a not `Thing` value`"),
        }
    }
    
    pub fn as_thing_mut(&mut self) -> (usize, &mut V) {
        match self {
            Self::Thing(t) => (t.0, &mut t.1),
            _ => panic!("Called `RawField::as_thing_mut()` on a not `Thing` value`"),
        }
    }
    
    /// 得到Thing内部值
    ///
    /// # Panics
    /// 非Thing时panic
    pub fn unwrap(self) -> V {
        match self {
            Self::Thing(t) => t.1,
            _ => panic!("Called `RawField::unwrap()` on a not `Thing` value`"),
        }
    }
    
    pub fn void() -> Self {
        Self::Void
    }
    
    /// 从Thing制造Prev
    ///
    /// # Panics
    /// 非Thing时Panic
    pub fn make_prev(&self) -> Self {
        match self {
            Self::Thing(t) => Self::Prev(t.0),
            _ => panic!("Called `RawField::make_prev()` on a not `Thing` value`"),
        }
    }
    
    /// 从Thing制造Next
    ///
    /// # Panics
    /// 非Thing时Panic
    pub fn make_next(&self) -> Self {
        match self {
            Self::Thing(t) => Self::Next(t.0),
            _ => panic!("Called `RawField::make_next()` on a not `Thing` value`"),
        }
    }
    
    pub fn prev_from(tuple: &(usize, V)) -> Self {
        Self::Prev(tuple.0)
    }
    
    pub fn next_from(tuple: &(usize, V)) -> Self {
        Self::Next(tuple.0)
    }
    
    
    /// 得到当前或上一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有前一个非空块，返回该块 <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    pub fn thing_prev(&self) -> Option<usize> {
        match self {
            RawField::Thing(v) => Some(v.0),
            RawField::Prev(prev)
            | RawField::Among(prev,..)
            => Some(*prev),
            _ => None,
        }
    }
    
    
    /// 得到当前或下一个非空块的索引
    ///
    /// 若块不为空，返回自己 <br>
    /// 若块为空且有后一个非空块，返回该块 <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    pub fn thing_next(&self) -> Option<usize> {
        match self {
            RawField::Thing(v) => Some(v.0),
            RawField::Next(next)
            | RawField::Among(_, next)
            => Some(*next),
            _ => None,
        }
    }
    
    pub fn partial_clone(&self) -> Option<Self> {
        match *self {
            RawField::Thing(_) => None,
            _ => Some(match *self {
                RawField::Prev(p) => RawField::Prev(p),
                RawField::Among(p, n) => RawField::Among(p, n),
                RawField::Next(n) => RawField::Next(n),
                RawField::Void => RawField::Void,
                RawField::Thing(_) => unreachable!(),
            }),
        }
    }
    
    /// 得到当前或上一个非空块的索引，使用变体区分
    ///
    /// 若块不为空，返回Some(Ok) <br>
    /// 若块为空且有前一个非空块，返回Some(Err) <br>
    /// 若块为空且没有前一个非空块，返回None <br>
    pub fn thing_or_prev(&self) -> Option<Result<usize,usize>> {
        match self {
            RawField::Thing(v) => Some(Ok(v.0)),
            RawField::Prev(prev)
            | RawField::Among(prev,..)
            => Some(Err(*prev)),
            _ => None,
        }
    }
    
    /// 得到当前或下一个非空块的索引，使用变体区分
    ///
    /// 若块不为空，返回Some(Ok) <br>
    /// 若块为空且有后一个非空块，返回Some(Err) <br>
    /// 若块为空且没有后一个非空块，返回None <br>
    pub fn thing_or_next(&self) -> Option<Result<usize,usize>> {
        match self {
            RawField::Thing(v) => Some(Ok(v.0)),
            RawField::Next(next)
            | RawField::Among(_, next)
            => Some(Err(*next)),
            _ => None,
        }
    }
    
}

impl<V> Clone for RawField<V> {
    /// # Panics
    /// 当RawField为Thing变体时panic，因为Thing不支持克隆！
    fn clone(&self) -> Self {
        self.partial_clone().expect("Called `RawField::clone` on a `Thing` value")
    }
}

#[derive(Debug)]
pub enum Field<V,C>{
    Elem(V),
    Collex(C)
}

impl<E,V> FieldItem<E> for Field<E,FieldCollex<E,V>>
where
    E: Collexetable<V>,
    V: FieldValue,
{
    fn first(&self) -> &E {
        match self{
            Field::Elem(e) => {e}
            Field::Collex(collex) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                collex.first().unwrap()
            }
        }
    }
    
    fn last(&self) -> &E {
        match self{
            Field::Elem(e) => {e},
            Field::Collex(collex) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                collex.last().unwrap()
            }
        }
    }
}

impl<V> FieldItem<V> for Field<V,FieldSet<V>>
where
    V: FieldValue,
{
    fn first(&self) -> &V {
        match self{
            Field::Elem(e) => {e}
            Field::Collex(set) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                set.first_in().unwrap()
            }
        }
    }
    
    fn last(&self) -> &V {
        match self{
            Field::Elem(e) => {e},
            Field::Collex(set) => {
                // 递归结构是所有权关系，不可能导致死循环。
                // 只有为空时才会None，而空时不会置为Thing
                set.last_in().unwrap()
            }
        }
    }
    
}

