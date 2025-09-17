use morphology_runtime::api::MorphologyRuntime;

fn main() {
    let runtime = MorphologyRuntime::new_from_resources("resources/model.onnx", "resources")
        .expect("failed to init runtime");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: analyze <token> [<token> ...]");
        std::process::exit(1);
    }
    let analyses = runtime.analyze(args).expect("analyze failed");
    for a in analyses {
        println!("{}\t{}\t{}", a.token, a.tag, a.lemma);
    }
}


