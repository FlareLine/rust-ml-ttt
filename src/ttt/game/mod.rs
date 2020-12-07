use crate::ttt::board::Board;
use crate::ttt::state::State;

pub struct Game {
  pub board: Board,
  pub state: State,
}

impl Game {
  pub fn initialize(mut self) {
    self.board = Board::default();
    self.state = State::InProgress;
  }
}
