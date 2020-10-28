use std::rc::Rc;
use crate::nn::connection::Connection;
use crate::nn::math::Numeric;

#[derive(Default)]
pub struct Neuron {
  pub inputs: Vec<Rc<Connection>>,
  pub outputs: Vec<Rc<Connection>>,
  pub value: f32,
  pub bias: f32,
  pub stale: bool,
}

impl Neuron {
  pub fn calculate(&mut self) -> f32 {
    if self.stale {
      let total: f32 = self.inputs.iter_mut().map(|i| Rc::get_mut(i).unwrap().calculate()).sum::<f32>() - self.bias;
      self.value = total.sigmoid();
      self.stale = false;
    }
    self.value
  }

  pub fn stale(&mut self) {
    self.stale = true;
    for output in &mut self.outputs {
      Rc::get_mut(output).unwrap().stale();
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  pub fn stalefn_makesneuron_stale() {
    let mut neuron: Neuron = Neuron {
      stale: false,
      ..Neuron::default()
    };
    neuron.stale();
    assert_eq!(neuron.stale, true, "Neuron.stale() did not make neuron stale.");
  }

  #[test]
  pub fn calculatefn_returnsexistingvaluefrom_freshneuron() {
    let mut neuron: Neuron = Neuron {
      value: 5.0,
      ..Neuron::default()
    };
    assert_eq!(neuron.calculate(), 5.0, "Neuron.calculate() did not return fresh value.");
  }

  #[test]
  pub fn calculatefn_returnsdefaultvaluefrom_staleneuronwithnoinputs() {
    let mut neuron: Neuron = Neuron {
      stale: true,
      value: 5.0,
      ..Neuron::default()
    };
    assert_eq!(neuron.calculate(), 0.5, "Neuron.calculate() with no inputs did not return default value.");
  }
}
