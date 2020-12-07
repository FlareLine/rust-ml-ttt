use std::rc::{Rc, Weak};
use std::cell::RefCell;
use crate::nn::connection::Connection;
use crate::nn::math::Numeric;

#[derive(Default)]
pub struct Neuron {
  pub inputs: Vec<Weak<RefCell<Connection>>>,
  pub outputs: Vec<Rc<RefCell<Connection>>>,
  pub value: RefCell<f32>,
  pub bias: RefCell<f32>,
  pub stale: RefCell<bool>,
}

impl Neuron {
  pub fn calculate(&self) -> f32 {
    let mut is_stale = self.stale.borrow_mut();
    let mut value = self.value.borrow_mut();
    if *is_stale {
      let inputs = self.inputs.iter();
      let valid_inputs = inputs.filter_map(Weak::upgrade);
      let total: f32 = valid_inputs.map(|i| i.borrow().calculate()).sum::<f32>() - *self.bias.borrow();
      *value = total.sigmoid();
      *is_stale = false;
    }
    *value
  }

  pub fn stale(&self) {
    *self.stale.borrow_mut() = true;
    for output in &self.outputs {
      output.borrow_mut().stale();
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  pub fn stalefn_makesneuron_stale() {
    let neuron: Neuron = Neuron {
      stale: false.into(),
      ..Neuron::default()
    };
    neuron.stale();
    assert_eq!(*neuron.stale.borrow(), true, "Neuron.stale() did not make neuron stale.");
  }

  #[test]
  pub fn calculatefn_returnsexistingvaluefrom_freshneuron() {
    let neuron: Neuron = Neuron {
      value: 5.0.into(),
      ..Neuron::default()
    };
    assert_eq!(neuron.calculate(), 5.0, "Neuron.calculate() did not return fresh value.");
  }

  #[test]
  pub fn calculatefn_returnsdefaultvaluefrom_staleneuronwithnoinputs() {
    let neuron: Neuron = Neuron {
      stale: true.into(),
      value: 5.0.into(),
      ..Neuron::default()
    };
    assert_eq!(neuron.calculate(), 0.5, "Neuron.calculate() with no inputs did not return default value.");
  }
}
