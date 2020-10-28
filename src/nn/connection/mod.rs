use crate::nn::neuron::Neuron;

pub struct Connection {
  pub input: Neuron,
  pub output: Neuron,
  pub weight: f32,
}

impl Connection {
  pub fn calculate(&mut self) -> f32 {
    self.input.calculate() * self.weight
  }

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
      stale: false,
      ..Neuron::default()
    };

    let mut connection: Connection = Connection {
      output: neuron,
      ..Connection::default()
    };

    connection.stale();
    assert_eq!(connection.output.stale, true, "Connection.stale() did not make output neuron stale.");
  }

  #[test]
  pub fn calculatefn_withdefaultweight_returnsinputvalue() {
    let neuron: Neuron = Neuron {
      stale: false,
      value: 5.0,
      ..Neuron::default()
    };

    let mut connection: Connection = Connection {
      input: neuron,
      ..Connection::default()
    };

    assert_eq!(connection.calculate(), 5.0, "Connection.calculate() with default weight did not return neuron value.");
  }

  #[test]
  pub fn calculatefn_withweight_returnsweightedvalue() {
    let neuron: Neuron = Neuron {
      stale: false,
      value: 5.0,
      ..Neuron::default()
    };

    let mut connection: Connection = Connection {
      input: neuron,
      weight: 2.0,
      ..Connection::default()
    };

    assert_eq!(connection.calculate(), 10.0, "Connection.calculate() with weight did not return weighted value.");
  }
}
