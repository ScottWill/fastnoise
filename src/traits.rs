pub trait Builder {
    type Output;
    fn build(self) -> Self::Output;
}

pub trait Sampler {
    fn sample3d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec3A>;
    fn sample2d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec2> {
        self.sample3d(position.into().extend(0.0))
    }
}
