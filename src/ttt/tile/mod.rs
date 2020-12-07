pub enum Tile {
  Empty,
  Cross,
  Naught,
}

impl ToString for Tile {
  fn to_string(&self) -> String {
    match self {
      Tile::Cross => String::from("X"),
      Tile::Naught => String::from("O"),
      Tile::Empty => String::from(" "),
    }
  }
}
