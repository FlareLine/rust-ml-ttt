/// Connection model for neural network.

use crate::nn::neuron::Neuron;

/// Connection between two neurons.
pub struct Connection {
  /// The input neuron.
  pub input: Neuron,
  /// The output neuron.
  pub output: Neuron,
  /// The weight value for this connection.
  pub weight: f32,
}

impl Connection {
  /// Calculate the effective value for this connection.
  pub fn calculate(&self) -> f32 {
    self.input.calculate() * self.weight
  }

  /// Make the output neuron for this connection stale.
  pub fn stale(&mut self) {
    self.output.stale()
  }
}

impl Default for Connection {
  fn default() -> Self {
    Self {
      input: Neuron::default(),
      output: Neuron::default(),
      weight: 1.0,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  pub fn stalefn_makesoutputneuron_stale() {
    let neuron: Neuron = Neuron {
      stale: false.into(),
      ..Neuron::default()
    };

    let mut connection: Connection = Connection {
      output: neuron,
      ..Connection::default()
    };

    connection.stale();
    assert_eq!(*connection.output.stale.borrow(), true, "Connection.stale() did not make output neuron stale.");
  }

  #[test]
  pub fn calculatefn_withdefaultweight_returnsinputvalue() {
    let neuron: Neuron = Neuron {
      stale: false.into(),
      value: 5.0.into(),
      ..Neuron::default()
    };

    let connection: Connection = Connection {
      input: neuron,
      ..Connection::default()
    };

    assert_eq!(connection.calculate(), 5.0, "Connection.calculate() with default weight did not return neuron value.");
  }

  #[test]
  pub fn calculatefn_withweight_returnsweightedvalue() {
    let neuron: Neuron = Neuron {
      stale: false.into(),
      value: 5.0.into(),
      ..Neuron::default()
    };

    let connection: Connection = Connection {
      input: neuron,
      weight: 2.0,
      ..Connection::default()
    };

    assert_eq!(connection.calculate(), 10.0, "Connection.calculate() with weight did not return weighted value.");
  }
}
