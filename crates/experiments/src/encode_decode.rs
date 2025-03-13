use std::time::Instant;

use clap::Parser;
use lz78::{
    encoder::{Encoder, LZ8Encoder},
    sequence::{CharacterSequence, Sequence},
};
use lz78_experiments::{argparse::EncodeDecodeCli, utils::read_file_to_string};

fn main() {
    let cli = EncodeDecodeCli::parse();
    let data = read_file_to_string(&cli.filename).expect("file read error");
    let lz78 = LZ8Encoder::new();
    let input = CharacterSequence::from_data_inferred_character_map(data);

    println!("Starting encoding");
    let tic = Instant::now();
    let encoded = lz78.encode(&input).expect("encoding error");
    let enc_time = tic.elapsed().as_secs_f32();
    println!("Encoded bytes: {}", encoded.compressed_len_bytes());
    println!("Encoding time (s): {enc_time}");
    println!(
        "Encoding time (ns/byte): {}",
        enc_time * 1e9 / (input.len() as f32)
    );
    println!("Compression ratio: {}", encoded.compression_ratio());

    let tic = Instant::now();
    let mut output = CharacterSequence::new(input.character_map.clone());
    lz78.decode(&mut output, &encoded).expect("decode error");
    let dec_time = tic.elapsed().as_secs_f32();

    println!("Decoding time (s): {dec_time}");
    println!(
        "Decoding time (ns/byte): {}",
        dec_time * 1e9 / (input.len() as f32)
    );
}
