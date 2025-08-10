use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ParameterWithGrad<TW, TG> {
    pub weight: TW,
    pub gradient: TG,
}

impl<TW, TG> ParameterWithGrad<TW, TG>
where
    TG: Default,
{
    pub fn new(weight: TW) -> Self {
        Self {
            weight,
            gradient: TG::default(),
        }
    }

    pub fn with_grad(weight: TW, gradient: TG) -> Self {
        Self { weight, gradient }
    }

    pub fn zero_grad(&mut self) {
        self.gradient = TG::default();
    }
}