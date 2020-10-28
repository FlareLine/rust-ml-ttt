use crate::ml::node::Node;

pub struct Neuron {
  pub next: Node,
  pub strength: f32,


impl Neuron 
  pub fn fire(&self, value: f32) -> f32 {
    let val : f32 = value * &self.strength;
    &self.next.fire(val)
  }
}
