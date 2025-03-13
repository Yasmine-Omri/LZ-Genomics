use clap::{Parser, ValueEnum};

/// Build an LZ78 SPA
#[derive(Parser)]
pub struct TrainCli {
    /// Path to save the file
    #[arg(long, short)]
    pub save_path: String,

    /// Directory where all data files are stored
    #[arg(long, short)]
    pub data_dir: String,

    /// Which dataset to use
    #[arg(long, short)]
    pub dataset: Datasets,

    /// Number of times to repeat the input data
    #[arg(long, default_value_t = 1)]
    pub repeat: u32,

    /// Number of samples to take. Defaults to the whole dataset.
    #[arg(long)]
    pub samples: Option<u64>,

    /// Whether to start from the root of the LZ78 tree for each new sample
    #[arg(long, default_value_t = false)]
    pub start_at_root: bool,

    /// LZ78 SPA smoothing parameter
    #[arg(long, default_value_t = 0.5)]
    pub gamma: f64,
}

/// Generate from an LZ78 SPA built from train.rs
#[derive(Parser)]
pub struct GenerateCli {
    /// Path where the SPA was saved
    #[arg(long, short)]
    pub save_path: String,

    /// Training dataset
    #[arg(long, short)]
    pub dataset: Datasets,
    /// Length of sequence to generate
    #[arg(short, default_value_t = 2000)]
    pub n: u64,

    /// Temperature of the generator
    #[arg(long, short, default_value_t = 0.1)]
    pub temperature: f64,

    /// Topk parameter of the generator
    #[arg(long, default_value_t = 5)]
    pub topk: u32,

    /// Minimum desired context in the LZ78 tree
    #[arg(long, default_value_t = 500)]
    pub min_context: u64,

    /// String with which to seed the generator
    #[arg(long)]
    pub seed_data: Option<String>,
}

/// LZ78-encode and then decode a file
#[derive(Parser)]
pub struct EncodeDecodeCli {
    #[arg(long, short)]
    pub filename: String,
}

/// Use the LZ78 SPA for image classification
#[derive(Parser)]
pub struct ImageClassificationCli {
    /// Directory where all data files are stored
    #[arg(long, short)]
    pub data_dir: String,

    /// Which dataset to use
    #[arg(long, short)]
    pub dataset: Datasets,

    /// Number of times to repeat the input data
    #[arg(long, default_value_t = 1)]
    pub repeat: u32,

    /// LZ78 SPA smoothing parameter
    #[arg(long, default_value_t = 0.5)]
    pub gamma: f64,

    /// Degree of quantization of pixel values, between 1 and 128
    #[arg(long, default_value_t = 1)]
    pub quant_strength: u8,
}

/// Use the LZ78 SPA for compression, inspired by (Merhav 2022):
/// https://arxiv.org/pdf/2212.12208
#[derive(Parser)]
pub struct SourceCompressionCli {
    /// How to generate the data to be compressed
    #[arg(long, short)]
    pub data_generator: DataGenerators,

    /// For a Bernoulli source, the Bernoulli parameter
    #[arg(long, short, default_value_t = 0.1)]
    pub prob_one: f64,

    /// Maximum sequence length
    #[arg(long, short, default_value_t = 50)]
    pub k_max: u64,

    /// Number of times to repeat the experiment
    #[arg(long, default_value_t = 1)]
    pub trials: u64,

    /// LZ78 SPA smoothing parameter
    #[arg(long, default_value_t = 0.5)]
    pub gamma: f64,

    /// Maximum number of threads to spawn
    #[arg(long, default_value_t = 8)]
    pub max_thread: u64,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum Datasets {
    #[value()]
    FashionMnist,

    #[value()]
    Mnist,

    #[value()]
    Cifar10,

    #[value()]
    Wikitext,

    #[value()]
    C4,

    #[value()]
    Imdb,

    #[value()]
    Spam,

    #[value()]
    Shakespeare,

    #[value()]
    TinyStories,
}

#[derive(ValueEnum, Clone, Copy)]
pub enum DataGenerators {
    #[value()]
    Bernoulli,

    #[value()]
    BernoulliLZ78Source,
}
