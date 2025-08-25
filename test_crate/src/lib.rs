/// A test struct with example code
/// 
/// # Example
/// ```rust
/// use test_examples::TestStruct;
/// 
/// let test = TestStruct::new("example");
/// println!("{}", test.value());
/// ```
pub struct TestStruct {
    value: String,
}

impl TestStruct {
    /// Create a new TestStruct
    /// 
    /// # Example
    /// ```rust
    /// let test = TestStruct::new("hello");
    /// assert_eq!(test.value(), "hello");
    /// ```
    pub fn new(value: &str) -> Self {
        TestStruct {
            value: value.to_string(),
        }
    }
    
    /// Get the value
    /// 
    /// # Example
    /// ```rust
    /// let test = TestStruct::new("world");
    /// let val = test.value();
    /// println!("Value: {}", val);
    /// ```
    pub fn value(&self) -> &str {
        &self.value
    }
}