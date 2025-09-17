use morphology_runtime::api::MorphologyRuntime;

fn main() {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = format!("{}/resources/model.onnx", crate_dir);
    let res_dir = format!("{}/resources", crate_dir);
    let runtime = MorphologyRuntime::new_from_resources(&model_path, &res_dir)
        .expect("failed to init runtime");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        use std::io::{self, Read};
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).expect("read stdin");
        for line in buf.lines() {
            let toks: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if toks.is_empty() { continue; }
            let analyses = runtime.analyze(toks).expect("analyze failed");
            for a in analyses { println!("{}\t{}\t{}", a.token, a.tag, a.lemma); }
        }
    } else {
        let analyses = runtime.analyze(args).expect("analyze failed");
        for a in analyses { println!("{}\t{}\t{}", a.token, a.tag, a.lemma); }
    }
}


