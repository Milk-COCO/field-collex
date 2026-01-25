use super::map::RawFieldMap;
use std::ops::{Div, Sub};
use num_traits::real::Real;
use num_traits::Zero;

pub struct RawFieldSet<K>
where
    K: Default + Copy + Ord + Div<K,Output=K> + Sub<K,Output=K> + TryInto<usize> + Zero + Real + Sized,
{
    map: RawFieldMap<K,()>,
}

