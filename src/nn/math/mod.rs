/// Maths utilities.

use std::f32::consts::E;

/// Numeric extension trait.
pub trait Numeric {
  /// Calculate the signmoid function for the current numeric value.
  fn sigmoid(&self) -> f32;
}

impl Numeric for f32 {
  fn sigmoid(&self) -> f32 {
    1.0 / (1.0 + E.powf(self * -1.0))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zerosigmoid_isequalto_pointfive() {
    let value: f32 = 0.0;
    let expected: f32 = 0.5;
    assert_eq!(value.sigmoid(), expected, "sigmoid(0.0) != 0.5");
  }

  #[test]
  fn positivesigmoid_isgreaterthan_negativesigmoid() {
    let postive: f32 = 10.0;
    let negative: f32 = -10.0;
    assert!(postive.sigmoid() > negative.sigmoid(), "sigmoid(+N) <= sigmoid(-N)");
  }
}
