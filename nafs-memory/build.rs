fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/barq_db.proto", "proto/barq_graph.proto"],
            &["proto"],
        )?;
    Ok(())
}
