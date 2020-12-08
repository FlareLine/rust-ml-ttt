/// Neuron model for neural network.

use std::rc::{Rc, Weak};
use std::cell::RefCell;
use crate::nn::connection::Connection;
use crate::nn::math::Numeric;

/// Single neuron with connections, a bias and a value.
#[derive(Default)]
pub struct Neuron {
  /// Input connections to this neuron.
  pub inputs: Vec<Weak<RefCell<Connection>>>,

  /// Output connections for this neuron.
  pub outputs: Vec<Rc<RefCell<Connection>>>,

  /// The neuron's current value.
  pub value: RefCell<f32>,

  /// This neuron's bias value, if any.
  pub bias: RefCell<Option<f32>>,

  /// Whether or not this neuron's value is stale and should be refreshed.
  pub stale: RefCell<bool>,

}

impl Neuron {
  /// Calculate this neuron's value based on input connections.
  pub fn calculate(&self) -> f32 {
    let mut is_stale = self.stale.borrow_mut();
    let mut value = self.value.borrow_mut();
    if *is_stale {
      let inputs = self.inputs.iter();
      let valid_inputs = inputs.filter_map(Weak::upgrade);
      let total: f32 = valid_inputs.map(|i| i.borrow().calculate()).sum::<f32>() - (*self.bias.borrow()).unwrap_or(1.0);
      *value = total.sigmoid();
      *is_stale = false;
    }
    *value
  }

  /// Mark this neuron and any output connections as stale.
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
