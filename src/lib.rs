pub mod raw;
pub mod recurs;

pub(crate) trait FieldItem<V:Copy> {
    fn first(&self) -> V;
    fn last(&self) -> V;
}

